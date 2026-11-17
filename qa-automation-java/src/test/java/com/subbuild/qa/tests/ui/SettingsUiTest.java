package com.subbuild.qa.tests.ui;

import com.subbuild.qa.base.BaseTest;
import com.subbuild.qa.pages.SettingsPage;
import io.restassured.response.Response;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Settings, Localization (RTL/LTR), Theme & Cost Estimator UI Tests")
class SettingsUiTest extends BaseTest {

    private Map<String, Object> originalSettings;

    @BeforeEach
    void snapshotSettings() {
        originalSettings = api.getSettings().jsonPath().getMap("settings");
    }

    @AfterEach
    void restoreSettings() {
        if (originalSettings != null) {
            api.updateSettings(originalSettings);
        }
    }

    @Test
    @Tag("smoke")
    @Tag("regression")
    @Tag("ui")
    @DisplayName("Language toggle flips RTL/LTR direction, theme switches dark/light, and AI toggles persist")
    void languageThemeAndTogglesPersist() {
        SettingsPage settings = openStudio().sidebar().openSettings();

        // Language & direction check
        settings.switchToHebrew();
        assertEquals("rtl", settings.getDocumentDir());
        assertEquals("he", settings.getDocumentLang());

        settings.switchToEnglish();
        assertEquals("ltr", settings.getDocumentDir());
        assertEquals("en", settings.getDocumentLang());

        // Theme switch check
        settings.switchToLightTheme();
        assertEquals("light", settings.getDocumentTheme());
        settings.switchToDarkTheme();
        assertEquals("dark", settings.getDocumentTheme());

        // Cost estimator table renders
        assertTrue(settings.isCostEstimateTableVisible());

        // Toggle Metric conversion + Cost-Saver mode + Save
        boolean metricBefore = settings.isMetricConversionEnabled();
        boolean saverBefore = settings.isCostSaverEnabled();
        settings.toggleMetricConversion()
                .toggleCostSaver()
                .saveSettings();

        Response persisted = api.getSettings();
        assertEquals(!metricBefore, persisted.jsonPath().getBoolean("settings.convert_to_metric"));
        assertEquals(!saverBefore, persisted.jsonPath().getBoolean("settings.cost_saver_mode"));

        // Navigate back to Ingest screen cleanly
        settings.backToStudio();
    }

    @Test
    @Tag("regression")
    @Tag("ui")
    @DisplayName("Enabling Flex tier and default on-screen position persists via Settings API")
    void flexAndOnscreenPositionPersist() {
        SettingsPage settings = openStudio().sidebar().openSettings();

        boolean flexBefore = settings.isTranslationFlexEnabled();
        settings.toggleTranslationFlex()
                .selectDefaultOnscreenPosition("middle_right")
                .saveSettings();

        Response persisted = api.getSettings();
        assertEquals(!flexBefore, persisted.jsonPath().getBoolean("settings.translation.flex"));
        assertEquals("middle_right", persisted.jsonPath().getString("settings.onscreen_position"));
    }
}
