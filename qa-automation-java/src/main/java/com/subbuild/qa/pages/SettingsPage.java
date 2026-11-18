package com.subbuild.qa.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.Select;

import java.time.Duration;
import java.util.List;

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

    private static final By APIKEY_TOGGLE = By.cssSelector("[data-testid='settings-apikey-toggle']");
    private static final By APIKEY_LINK = By.cssSelector("[data-testid='settings-apikey-link']");
    private static final By APIKEY_LABEL_INPUT = By.cssSelector("[data-testid='provider-key-label-input']");
    private static final By APIKEY_SECRET_INPUT = By.cssSelector("[data-testid='settings-apikey-input']");
    private static final By APIKEY_SAVE_BTN = By.cssSelector("[data-testid='settings-apikey-save']");
    private static final By APIKEY_LIST = By.cssSelector("[data-testid='provider-key-list']");
    private static final By OLLAMA_PRESET_CLOUD = By.cssSelector("[data-testid='ollama-preset-cloud']");
    private static final By OLLAMA_BASE_URL_INPUT = By.cssSelector("[data-testid='provider-key-baseurl-input']");
    private static final By APPLY_PROVIDER_ALL_BTN = By.cssSelector("[data-testid='apply-provider-all-btn']");

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

    public SettingsPage expandProviderKeys() {
        if (!isDisplayedQuick(APIKEY_LIST, Duration.ofSeconds(1))) {
            click(APIKEY_TOGGLE);
            waitVisible(APIKEY_LIST);
        }
        return this;
    }

    public SettingsPage selectProviderTab(String provider) {
        expandProviderKeys();
        click(By.cssSelector("[data-testid='provider-tab-" + provider + "']"));
        return this;
    }

    public SettingsPage addProviderKey(String provider, String label, String secret) {
        selectProviderTab(provider);
        int before = getProviderKeyItemsCount();
        clearAndType(APIKEY_LABEL_INPUT, label);
        clearAndType(APIKEY_SECRET_INPUT, secret);
        click(APIKEY_SAVE_BTN);
        wait.until(d -> getProviderKeyItemsCount() > before);
        return this;
    }

    public int getProviderKeyItemsCount() {
        return driver.findElements(By.cssSelector("[data-testid^='provider-key-item-']")).size();
    }

    public String getProviderKeysText() {
        return waitVisible(APIKEY_LIST).getText();
    }

    public String getProviderKeyLinkHref() {
        return waitVisible(APIKEY_LINK).getAttribute("href");
    }

    public SettingsPage clickOllamaCloudPreset() {
        selectProviderTab("ollama");
        click(OLLAMA_PRESET_CLOUD);
        wait.until(d -> "https://ollama.com".equals(waitVisible(OLLAMA_BASE_URL_INPUT).getAttribute("value")));
        return this;
    }

    public String getOllamaBaseUrlValue() {
        return waitVisible(OLLAMA_BASE_URL_INPUT).getAttribute("value");
    }

    public SettingsPage selectStepProvider(String step, String provider) {
        By sel = By.cssSelector("[data-testid='model-card-" + step + "-provider']");
        scrollIntoView(waitVisible(sel));
        selectByValue(sel, provider);
        return this;
    }

    public SettingsPage selectStepModel(String step, String modelId) {
        By sel = By.cssSelector("[data-testid='model-card-" + step + "-model']");
        scrollIntoView(waitVisible(sel));
        selectByValue(sel, modelId);
        return this;
    }

    public String getSelectedStepProvider(String step) {
        return selectedValue(By.cssSelector("[data-testid='model-card-" + step + "-provider']"));
    }

    public String getSelectedStepModel(String step) {
        return selectedValue(By.cssSelector("[data-testid='model-card-" + step + "-model']"));
    }

    public SettingsPage clickApplyProviderToAll() {
        WebElement btn = waitClickable(APPLY_PROVIDER_ALL_BTN);
        scrollIntoView(btn);
        btn.click();
        return this;
    }

    public String getStepWarningText(String step) {
        return waitVisible(By.cssSelector("[data-testid='model-card-" + step + "-warning']")).getText();
    }

    public SettingsPage addOllamaEndpointKey(String label, String baseUrl, String secret) {
        selectProviderTab("ollama");
        int before = getProviderKeyItemsCount();
        clearAndType(APIKEY_LABEL_INPUT, label);
        clearAndType(OLLAMA_BASE_URL_INPUT, baseUrl);
        if (secret != null && !secret.isBlank()) {
            clearAndType(APIKEY_SECRET_INPUT, secret);
        }
        click(APIKEY_SAVE_BTN);
        wait.until(d -> getProviderKeyItemsCount() > before);
        return this;
    }

    public SettingsPage selectStepKeyByVisiblePrefix(String step, String labelPrefix) {
        By sel = By.cssSelector("[data-testid='model-card-" + step + "-key']");
        WebElement el = waitVisible(sel);
        scrollIntoView(el);
        Select select = new Select(el);
        for (WebElement opt : select.getOptions()) {
            if (opt.getText().contains(labelPrefix)) {
                select.selectByValue(opt.getAttribute("value"));
                return this;
            }
        }
        throw new IllegalArgumentException("Key option containing '" + labelPrefix + "' not found for step " + step);
    }

    public List<String> getStepModelOptionValues(String step) {
        By sel = By.cssSelector("[data-testid='model-card-" + step + "-model']");
        WebElement el = waitVisible(sel);
        return new Select(el).getOptions().stream()
                .map(o -> o.getAttribute("value"))
                .filter(v -> v != null && !v.isEmpty())
                .toList();
    }

    public List<String> getStepModelOptionLabels(String step) {
        By sel = By.cssSelector("[data-testid='model-card-" + step + "-model']");
        WebElement el = waitVisible(sel);
        return new Select(el).getOptions().stream()
                .map(WebElement::getText)
                .toList();
    }

    public boolean isStepNoModelsHintVisible(String step) {
        return isDisplayedQuick(By.cssSelector("[data-testid='model-card-" + step + "-no-models']"), Duration.ofSeconds(3));
    }

    public String getStepBenchmarksText(String step) {
        return waitVisible(By.cssSelector("[data-testid='model-card-" + step + "-benchmarks']")).getText();
    }

    public IngestPage backToStudio() {
        WebElement back = waitClickable(BACK_BTN);
        scrollIntoView(back);
        back.click();
        return new IngestPage(driver);
    }
}
