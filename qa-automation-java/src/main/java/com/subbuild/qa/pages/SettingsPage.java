package com.subbuild.qa.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;

import java.time.Duration;

/**
 * Page Object for the Settings screen (language, theme, AI model config, toggles, cost preview).
 */
public class SettingsPage extends BasePage {

    private static final By PAGE_CONTAINER = By.cssSelector("[data-testid='model-card-translation']");
    private static final By BACK_BTN = By.xpath("//button[.//*[contains(@class,'lucide-arrow-left')]]");

    private static final By LANG_HE_BTN = By.xpath("//div[@role='radiogroup']//button[contains(.,'עברית')]");
    private static final By LANG_EN_BTN = By.xpath("//div[@role='radiogroup']//button[contains(.,'English')]");
    private static final By THEME_LIGHT_BTN = By.xpath("//div[@role='radiogroup']//button[.//*[contains(@class,'lucide-sun')]]");
    private static final By THEME_DARK_BTN = By.xpath("//div[@role='radiogroup']//button[.//*[contains(@class,'lucide-moon')]]");

    private static final By METRIC_TOGGLE = By.cssSelector("[data-testid='metric-conversion-toggle']");
    private static final By METRIC_SWITCH = By.cssSelector("[data-testid='metric-conversion-switch']");
    private static final By COST_SAVER_TOGGLE = By.cssSelector("[data-testid='cost-saver-toggle']");
    private static final By COST_SAVER_SWITCH = By.cssSelector("[data-testid='cost-saver-switch']");
    private static final By TRANSLATION_FLEX_SWITCH = By.cssSelector("[data-testid='model-card-translation-flex']");

    private static final By SAVE_AI_SETTINGS_BTN = By.xpath(
            "//section[.//*[@data-testid='model-card-translation']]//button[.//*[contains(@class,'lucide-save') or contains(@class,'lucide-check')]]");
    private static final By ESTIMATE_TABLE = By.cssSelector("[data-testid='estimate-table']");

    public SettingsPage(WebDriver driver) {
        super(driver);
    }

    public SettingsPage waitForLoaded() {
        waitVisible(PAGE_CONTAINER);
        return this;
    }

    public SettingsPage switchToHebrew() {
        click(LANG_HE_BTN);
        wait.until(d -> "rtl".equals(getDocumentDir()));
        return this;
    }

    public SettingsPage switchToEnglish() {
        click(LANG_EN_BTN);
        wait.until(d -> "ltr".equals(getDocumentDir()));
        return this;
    }

    public SettingsPage switchToLightTheme() {
        click(THEME_LIGHT_BTN);
        wait.until(d -> "light".equals(getDocumentTheme()));
        return this;
    }

    public SettingsPage switchToDarkTheme() {
        click(THEME_DARK_BTN);
        wait.until(d -> "dark".equals(getDocumentTheme()));
        return this;
    }

    public SettingsPage toggleMetricConversion() {
        boolean before = isMetricConversionEnabled();
        WebElement toggle = waitVisible(METRIC_TOGGLE);
        scrollIntoView(toggle);
        toggle.click();
        wait.until(d -> isMetricConversionEnabled() != before);
        return this;
    }

    public boolean isMetricConversionEnabled() {
        return "true".equals(waitVisible(METRIC_SWITCH).getAttribute("aria-checked"));
    }

    public SettingsPage toggleCostSaver() {
        boolean before = isCostSaverEnabled();
        WebElement toggle = waitVisible(COST_SAVER_TOGGLE);
        scrollIntoView(toggle);
        toggle.click();
        wait.until(d -> isCostSaverEnabled() != before);
        return this;
    }

    public boolean isCostSaverEnabled() {
        return "true".equals(waitVisible(COST_SAVER_SWITCH).getAttribute("aria-checked"));
    }

    public SettingsPage toggleTranslationFlex() {
        boolean before = isTranslationFlexEnabled();
        WebElement el = waitVisible(TRANSLATION_FLEX_SWITCH);
        scrollIntoView(el);
        el.click();
        wait.until(d -> isTranslationFlexEnabled() != before);
        return this;
    }

    public boolean isTranslationFlexEnabled() {
        return waitVisible(TRANSLATION_FLEX_SWITCH).isSelected();
    }

    public SettingsPage selectDefaultOnscreenPosition(String position) {
        String icon = switch (position) {
            case "top_center" -> "⬆️";
            case "middle_right" -> "➡️";
            case "middle_left" -> "⬅️";
            case "top_right" -> "↗️";
            case "top_left" -> "↖️";
            default -> throw new IllegalArgumentException("Unknown onscreen position: " + position);
        };
        By posBtn = By.xpath(
                "//div[@data-testid='model-card-vision']//button[@type='button' and contains(.,'" + icon + "')]");
        WebElement btn = waitVisible(posBtn);
        scrollIntoView(btn);
        btn.click();
        return this;
    }

    public SettingsPage saveSettings() {
        WebElement saveBtn = waitClickable(SAVE_AI_SETTINGS_BTN);
        scrollIntoView(saveBtn);
        saveBtn.click();
        // Once persisted to backend and store, dirty becomes false and the button disables.
        wait.until(d -> !d.findElement(SAVE_AI_SETTINGS_BTN).isEnabled());
        return this;
    }

    public boolean isCostEstimateTableVisible() {
        return isDisplayedQuick(ESTIMATE_TABLE, Duration.ofSeconds(5));
    }

    public IngestPage backToStudio() {
        WebElement back = waitClickable(BACK_BTN);
        scrollIntoView(back);
        back.click();
        return new IngestPage(driver);
    }
}
