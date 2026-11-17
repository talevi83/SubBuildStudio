package com.subbuild.qa.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;

/**
 * Page Component representing the Studio bottom ExportToolbar
 * (onscreen position selector, RTL toggle, SRT/VTT/Log exports, glossary, save next to video).
 */
public class ExportToolbarComponent extends BasePage {

    private static final By ONSCREEN_POS_SELECT = By.cssSelector("[data-testid='studio-onscreen-position']");
    private static final By EXPORT_ORIGINAL_SRT = By.cssSelector("[data-testid='export-original-srt']");
    private static final By EXPORT_VTT = By.cssSelector("[data-testid='export-vtt']");
    private static final By OPEN_GLOSSARY_BTN = By.cssSelector("[data-testid='open-glossary']");
    private static final By OPEN_EMBED_BTN = By.cssSelector("[data-testid='open-embed']");
    private static final By JOB_LOG_BTN = By.cssSelector("[data-testid='job-log']");
    private static final By SAVE_NEXT_TO_VIDEO_BTN = By.cssSelector("[data-testid='save-next-to-video']");

    public ExportToolbarComponent(WebDriver driver) {
        super(driver);
    }

    public ExportToolbarComponent waitVisible() {
        waitVisible(ONSCREEN_POS_SELECT);
        return this;
    }

    public String getOnscreenPosition() {
        return selectedValue(ONSCREEN_POS_SELECT);
    }

    public ExportToolbarComponent setOnscreenPosition(String position) {
        selectByValue(ONSCREEN_POS_SELECT, position);
        wait.until(d -> position.equals(selectedValue(ONSCREEN_POS_SELECT)));
        return this;
    }

    public boolean isOriginalSrtEnabled() {
        return waitVisible(EXPORT_ORIGINAL_SRT).isEnabled();
    }

    public boolean isWebVttEnabled() {
        return waitVisible(EXPORT_VTT).isEnabled();
    }

    public boolean isJobLogVisible() {
        return waitVisible(JOB_LOG_BTN).isDisplayed();
    }

    public boolean isGlossaryEnabled() {
        return waitVisible(OPEN_GLOSSARY_BTN).isEnabled();
    }

    public boolean isEmbedEnabled() {
        return waitVisible(OPEN_EMBED_BTN).isEnabled();
    }

    public boolean isSaveNextToVideoPresent() {
        return !driver.findElements(SAVE_NEXT_TO_VIDEO_BTN).isEmpty();
    }
}
