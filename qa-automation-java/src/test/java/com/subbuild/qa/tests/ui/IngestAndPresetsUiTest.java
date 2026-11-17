package com.subbuild.qa.tests.ui;

import com.subbuild.qa.base.BaseTest;
import com.subbuild.qa.pages.IngestPage;
import com.subbuild.qa.pages.StudioEditorPage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Ingest, Mode Switch & Genre Presets UI Tests")
class IngestAndPresetsUiTest extends BaseTest {

    @AfterEach
    void cleanupJobs() {
        api.deleteAllJobs();
    }

    @Test
    @Tag("smoke")
    @Tag("regression")
    @Tag("ui")
    @DisplayName("First-run API modal skip, ModeSwitch toggle, and local media probe with cost estimate")
    void ingestProbeAndModeSwitchFlow() {
        IngestPage ingest = openStudioRaw();
        assertTrue(ingest.isFirstRunModalVisible(), "First-run API key modal should appear when Gemini key is unset");
        ingest.dismissFirstRunModalIfPresent();

        assertEquals("transcribe", ingest.getSelectedMode());
        ingest.selectSyncMode();
        assertEquals("sync", ingest.getSelectedMode());
        ingest.selectTranscribeMode();
        assertEquals("transcribe", ingest.getSelectedMode());

        ingest.probeLocalPath(sampleVideoPath.toString());
        assertTrue(ingest.getProbeBannerText().contains("qa_sample.mkv"));
        assertTrue(ingest.isEmbeddedSubtitlesTogglePresent(), "Embedded dialogue track toggle should appear");
        assertTrue(ingest.isCostEstimateDisplayed(), "Cost estimate banner should be visible after probing");
    }

    @ParameterizedTest(name = "Genre preset \"{0}\" activates preset badge")
    @ValueSource(strings = {"drama", "docu", "action", "fast"})
    @Tag("regression")
    @Tag("ui")
    @DisplayName("Selecting a Genre Preset updates its active state and badge")
    void genrePresetsUpdateActiveBadge(String presetId) {
        IngestPage ingest = openStudio()
                .probeLocalPath(sampleVideoPath.toString())
                .applyGenrePreset(presetId);

        String badge = ingest.getActivePresetBadgeText();
        assertFalse(badge.isBlank());
        assertFalse(badge.toLowerCase().contains("custom"), "Preset badge should reflect selected preset, not Custom");
    }

    @Test
    @Tag("regression")
    @Tag("ui")
    @DisplayName("Launching an embedded-subtitle job from Ingest UI opens completed Studio workspace")
    void launchEmbeddedJobFromUi() {
        IngestPage ingest = openStudio().probeLocalPath(sampleVideoPath.toString());
        StudioEditorPage studio = ingest.launchJob().waitForCompletedStatus();

        assertEquals("qa_sample.mkv", studio.getHeaderTitle());
        assertEquals(3, studio.getVisibleSubtitleRowCount());
        assertEquals("Hello from SubBuild QA.", studio.getOriginalTextAtRow(0));
    }
}
