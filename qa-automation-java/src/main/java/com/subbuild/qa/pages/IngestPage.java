package com.subbuild.qa.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;

import java.time.Duration;

/**
 * Page Object for SubBuild Studio's Ingest & Pipeline Configuration screen.
 */
public class IngestPage extends BasePage {

    private static final By FIRST_RUN_MODAL = By.cssSelector("[data-testid='first-run-apikey-modal']");
    private static final By FIRST_RUN_SKIP_BTN = By.cssSelector("[data-testid='modal-apikey-skip']");

    private static final By MODE_TRANSCRIBE = By.cssSelector("[data-testid='mode-transcribe']");
    private static final By MODE_SYNC = By.cssSelector("[data-testid='mode-sync']");

    private static final By LOCAL_PATH_INPUT = By.cssSelector("[data-testid='local-path-input']");
    private static final By INSPECT_SUBMIT_BTN = By.xpath("//input[@data-testid='local-path-input']/ancestor::form//button[@type='submit']");
    private static final By PROBE_SUCCESS_BANNER = By.cssSelector(".bg-emerald-500\\/10");
    private static final By PROBE_ERROR_BANNER = By.cssSelector(".bg-rose-500\\/10");

    private static final By SYNC_PICKER = By.cssSelector("[data-testid='subtitle-file-picker']");
    private static final By SYNC_FILE_INPUT = By.cssSelector("[data-testid='subtitle-file-picker'] input[type='file']");
    private static final By SYNC_ENCODING_SELECT = By.cssSelector("[data-testid='sync-encoding']");

    private static final By GENRE_PRESETS_SECTION = By.cssSelector("[data-testid='genre-presets']");
    private static final By ACTIVE_PRESET_BADGE = By.cssSelector("[data-testid='genre-presets'] span.rounded-full");
    private static final By ESTIMATE_CARD = By.cssSelector("[data-testid='file-estimate']");
    private static final By LAUNCH_BTN = By.cssSelector("[data-testid='launch']");

    private static final By USE_EMBEDDED_TOGGLE = By.cssSelector("[data-testid='use-embedded-subtitles-toggle']");

    public IngestPage(WebDriver driver) {
        super(driver);
    }

    public IngestPage openWithToken(String baseUrl, String token) {
        driver.get(baseUrl + "/?token=" + token);
        waitVisible(LOCAL_PATH_INPUT);
        return this;
    }

    public boolean isFirstRunModalVisible() {
        return isDisplayedQuick(FIRST_RUN_MODAL, Duration.ofSeconds(2));
    }

    public IngestPage dismissFirstRunModalIfPresent() {
        if (isFirstRunModalVisible()) {
            click(FIRST_RUN_SKIP_BTN);
            waitInvisible(FIRST_RUN_MODAL);
        }
        return this;
    }

    public IngestPage selectTranscribeMode() {
        click(MODE_TRANSCRIBE);
        return this;
    }

    public IngestPage selectSyncMode() {
        click(MODE_SYNC);
        return this;
    }

    public String getSelectedMode() {
        String syncChecked = waitVisible(MODE_SYNC).getAttribute("aria-checked");
        return "true".equals(syncChecked) ? "sync" : "transcribe";
    }

    public IngestPage probeLocalPath(String mediaPath) {
        clearAndType(LOCAL_PATH_INPUT, mediaPath);
        click(INSPECT_SUBMIT_BTN);
        waitVisible(PROBE_SUCCESS_BANNER);
        waitVisible(GENRE_PRESETS_SECTION);
        return this;
    }

    public String getProbeBannerText() {
        return waitVisible(PROBE_SUCCESS_BANNER).getText();
    }

    public String probeInvalidPathExpectError(String badPath) {
        clearAndType(LOCAL_PATH_INPUT, badPath);
        click(INSPECT_SUBMIT_BTN);
        return waitVisible(PROBE_ERROR_BANNER).getText();
    }

    public boolean isLaunchButtonPresent() {
        return !driver.findElements(LAUNCH_BTN).isEmpty();
    }

    public IngestPage uploadSyncSubtitleFile(String subtitleFilePath) {
        waitVisible(SYNC_PICKER);
        WebElement fileInput = driver.findElement(SYNC_FILE_INPUT);
        fileInput.sendKeys(subtitleFilePath);
        waitVisible(SYNC_ENCODING_SELECT);
        return this;
    }

    public String getSyncEncodingValue() {
        return selectedValue(SYNC_ENCODING_SELECT);
    }

    public int getSyncPreviewCueCount() {
        return waitAllVisible(By.cssSelector("[data-testid='subtitle-file-picker'] ul li")).size();
    }

    public boolean isCostEstimateDisplayed() {
        return isDisplayedQuick(ESTIMATE_CARD, Duration.ofSeconds(5));
    }

    public IngestPage applyGenrePreset(String presetId) {
        By presetBtn = By.cssSelector("[data-testid='genre-preset-" + presetId + "']");
        WebElement btn = waitVisible(presetBtn);
        scrollIntoView(btn);
        wait.until(ExpectedConditions.elementToBeClickable(btn)).click();
        wait.until(d -> "true".equals(d.findElement(presetBtn).getAttribute("aria-pressed")));
        return this;
    }

    public String getActivePresetBadgeText() {
        return waitVisible(ACTIVE_PRESET_BADGE).getText();
    }

    public boolean isEmbeddedSubtitlesTogglePresent() {
        return isDisplayedQuick(USE_EMBEDDED_TOGGLE, Duration.ofSeconds(2));
    }

    public StudioEditorPage launchJob() {
        WebElement btn = waitClickable(LAUNCH_BTN);
        scrollIntoView(btn);
        btn.click();
        return new StudioEditorPage(driver).waitForWorkspaceLoaded();
    }

    public SidebarComponent sidebar() {
        return new SidebarComponent(driver);
    }
}
