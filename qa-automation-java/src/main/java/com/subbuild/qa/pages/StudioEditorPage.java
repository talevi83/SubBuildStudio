package com.subbuild.qa.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;

import java.time.Duration;
import java.util.List;

/**
 * Page Object for the Studio Editor workspace: SubtitleTable, Search & Replace,
 * Insert Segment modal, Undo/Redo History controls, QA Panel, and ExportToolbarComponent.
 */
public class StudioEditorPage extends BasePage {

    private static final By HEADER_TITLE = By.cssSelector("header span.text-sm.font-semibold");
    private static final By STATUS_BADGE = By.cssSelector("header div.text-xs.font-semibold");
    private static final By SUBTITLE_ROWS = By.cssSelector("tbody tr");

    private static final By ADD_LINE_BTN = By.cssSelector("[data-testid='add-line']");
    private static final By INSERT_DIALOG = By.cssSelector("[data-testid='insert-dialog']");
    private static final By INSERT_START_INPUT = By.cssSelector("[data-testid='insert-start']");
    private static final By INSERT_END_INPUT = By.cssSelector("[data-testid='insert-end']");
    private static final By INSERT_ORIGINAL_INPUT = By.cssSelector("[data-testid='insert-original']");
    private static final By INSERT_SUBMIT_BTN = By.cssSelector("[data-testid='insert-submit']");

    private static final By UNDO_BTN = By.cssSelector("[data-testid='history-undo']");
    private static final By REDO_BTN = By.cssSelector("[data-testid='history-redo']");
    private static final By HISTORY_TOGGLE_BTN = By.cssSelector("[data-testid='history-toggle']");
    private static final By HISTORY_PANEL = By.cssSelector("[data-testid='history-panel']");
    private static final By HISTORY_ENTRIES = By.cssSelector("[data-testid='history-entry']");

    private static final By QA_CHECK_BTN = By.cssSelector("[data-testid='qa-check']");
    private static final By QA_PANEL = By.cssSelector("[data-testid='qa-panel']");

    private static final By SHIFT_TIME_TOGGLE = By.cssSelector("[data-testid='shift-time-toggle']");
    private static final By SHIFT_TIME_POPOVER = By.cssSelector("[data-testid='shift-time-popover']");
    private static final By SHIFT_TIME_INPUT = By.cssSelector("[data-testid='shift-time-input']");
    private static final By SHIFT_TIME_APPLY = By.cssSelector("[data-testid='shift-time-apply']");

    private static final By INSERT_ISSUES = By.cssSelector("[data-testid='insert-issues']");
    private static final By INSERT_CANCEL_BTN = By.xpath("//*[@data-testid='insert-submit']/preceding-sibling::button[1]");

    private static final By OPEN_GLOSSARY_BTN = By.cssSelector("[data-testid='open-glossary']");
    private static final By GLOSSARY_PROFILE_INPUT = By.cssSelector("[data-testid='glossary-profile']");
    private static final By GLOSSARY_SAVE_BTN = By.cssSelector("[data-testid='glossary-save']");
    private static final By GLOSSARY_ADD_CHAR_BTN = By.xpath("//*[@data-testid='glossary-profile']/ancestor::div[contains(@class,'max-w-2xl')]//section[1]/button");

    private static final By SEARCH_INPUT = By.cssSelector("input[placeholder*='Search'], input[placeholder*='חיפוש']");
    private static final By REPLACE_INPUT = By.cssSelector("input[placeholder*='Replace'], input[placeholder*='החלפה']");
    private static final By REPLACE_COL_SELECT = By.xpath(
            "//input[contains(@placeholder,'Replace') or contains(@placeholder,'החלפה')]/ancestor::div[contains(@class,'flex items-center gap-2')][1]/select");
    private static final By REPLACE_ALL_BTN = By.xpath(
            "//input[contains(@placeholder,'Replace') or contains(@placeholder,'החלפה')]/ancestor::div[contains(@class,'flex items-center gap-2')][1]/button");

    public StudioEditorPage(WebDriver driver) {
        super(driver);
    }

    public StudioEditorPage waitForWorkspaceLoaded() {
        waitVisible(By.cssSelector("table"));
        exportToolbar().waitVisible();
        return this;
    }

    public StudioEditorPage waitForCompletedStatus() {
        wait.until(d -> {
            String text = waitVisible(STATUS_BADGE).getText().trim().toLowerCase();
            return text.contains("completed") || text.contains("הושלם");
        });
        waitVisible(ADD_LINE_BTN);
        return this;
    }

    public String getHeaderTitle() {
        return waitVisible(HEADER_TITLE).getText().trim();
    }

    public int getVisibleSubtitleRowCount() {
        waitAllVisible(SUBTITLE_ROWS);
        return driver.findElements(SUBTITLE_ROWS).size();
    }

    public String getOriginalTextAtRow(int zeroBasedIndex) {
        return wait.ignoring(org.openqa.selenium.StaleElementReferenceException.class, IndexOutOfBoundsException.class)
                .until(d -> {
                    List<WebElement> rows = d.findElements(SUBTITLE_ROWS);
                    if (rows.size() <= zeroBasedIndex) {
                        return null;
                    }
                    WebElement firstTextarea = rows.get(zeroBasedIndex).findElements(By.tagName("textarea")).get(0);
                    return firstTextarea.getAttribute("value");
                });
    }

    public StudioEditorPage insertSegment(String startTimecode, String endTimecode, String originalText) {
        int before = getVisibleSubtitleRowCount();
        click(ADD_LINE_BTN);
        waitVisible(INSERT_DIALOG);
        clearAndType(INSERT_START_INPUT, startTimecode);
        clearAndType(INSERT_END_INPUT, endTimecode);
        clearAndType(INSERT_ORIGINAL_INPUT, originalText);
        click(INSERT_SUBMIT_BTN);
        waitInvisible(INSERT_DIALOG);
        wait.until(d -> d.findElements(SUBTITLE_ROWS).size() == before + 1);
        return this;
    }

    public StudioEditorPage undo() {
        wait.until(d -> d.findElement(UNDO_BTN).isEnabled());
        click(UNDO_BTN);
        return this;
    }

    public StudioEditorPage redo() {
        wait.until(d -> d.findElement(REDO_BTN).isEnabled());
        click(REDO_BTN);
        return this;
    }

    public boolean isUndoEnabled() {
        try {
            new org.openqa.selenium.support.ui.WebDriverWait(driver, Duration.ofSeconds(5))
                    .until(d -> d.findElement(UNDO_BTN).isEnabled());
            return true;
        } catch (org.openqa.selenium.TimeoutException e) {
            return false;
        }
    }

    public boolean isRedoEnabled() {
        try {
            new org.openqa.selenium.support.ui.WebDriverWait(driver, Duration.ofSeconds(5))
                    .until(d -> d.findElement(REDO_BTN).isEnabled());
            return true;
        } catch (org.openqa.selenium.TimeoutException e) {
            return false;
        }
    }

    public int openHistoryAndCountEntries() {
        boolean openedHere = false;
        if (!isDisplayedQuick(HISTORY_PANEL, Duration.ofMillis(300))) {
            click(HISTORY_TOGGLE_BTN);
            waitVisible(HISTORY_PANEL);
            openedHere = true;
        }
        int count = waitAllVisible(HISTORY_ENTRIES).size();
        if (openedHere) {
            click(HISTORY_TOGGLE_BTN);
            waitInvisible(HISTORY_PANEL);
        }
        return count;
    }

    public StudioEditorPage replaceAllInOriginal(String findText, String replaceWith) {
        clearAndType(SEARCH_INPUT, findText);
        clearAndType(REPLACE_INPUT, replaceWith);
        selectByValue(REPLACE_COL_SELECT, "original");
        click(REPLACE_ALL_BTN);
        waitVisible(By.cssSelector("div.bg-sunken span.text-indigo-300"));
        clearAndType(SEARCH_INPUT, "");
        wait.until(d -> d.findElements(SUBTITLE_ROWS).size() > 1);
        return this;
    }

    public String runQualityCheck() {
        click(QA_CHECK_BTN);
        return waitVisible(QA_PANEL).getText();
    }

    public StudioEditorPage openInsertDialogWithOverlap(String startTimecode, String endTimecode, String text) {
        click(ADD_LINE_BTN);
        waitVisible(INSERT_DIALOG);
        clearAndType(INSERT_START_INPUT, startTimecode);
        clearAndType(INSERT_END_INPUT, endTimecode);
        clearAndType(INSERT_ORIGINAL_INPUT, text);
        waitVisible(INSERT_ISSUES);
        return this;
    }

    public String getInsertIssuesText() {
        return waitVisible(INSERT_ISSUES).getText();
    }

    public boolean isInsertSubmitEnabled() {
        return waitVisible(INSERT_SUBMIT_BTN).isEnabled();
    }

    public StudioEditorPage closeInsertDialog() {
        click(INSERT_CANCEL_BTN);
        waitInvisible(INSERT_DIALOG);
        return this;
    }

    public StudioEditorPage shiftAllTimes(String offsetSeconds) {
        click(SHIFT_TIME_TOGGLE);
        waitVisible(SHIFT_TIME_POPOVER);
        clearAndType(SHIFT_TIME_INPUT, offsetSeconds);
        click(SHIFT_TIME_APPLY);
        waitInvisible(SHIFT_TIME_POPOVER);
        // Wait until history records the shift_time step
        isUndoEnabled();
        return this;
    }

    public StudioEditorPage openGlossaryAndSaveCharacter(
            String charName, String targetName, String gender, String profileName) {
        click(OPEN_GLOSSARY_BTN);
        waitVisible(GLOSSARY_SAVE_BTN);
        click(GLOSSARY_ADD_CHAR_BTN);
        List<WebElement> inputs = waitAllVisible(
                By.xpath("//*[@data-testid='glossary-profile']/ancestor::div[contains(@class,'max-w-2xl')]//section[1]//input"));
        // New row added at the end (3 inputs per character row: name, target_name, notes)
        int base = inputs.size() - 3;
        inputs.get(base).sendKeys(charName);
        inputs.get(base + 1).sendKeys(targetName);
        List<WebElement> selects = waitAllVisible(
                By.xpath("//*[@data-testid='glossary-profile']/ancestor::div[contains(@class,'max-w-2xl')]//section[1]//select"));
        new org.openqa.selenium.support.ui.Select(selects.get(selects.size() - 1)).selectByValue(gender);
        clearAndType(GLOSSARY_PROFILE_INPUT, profileName);
        click(GLOSSARY_SAVE_BTN);
        waitInvisible(GLOSSARY_SAVE_BTN);
        return this;
    }

    public ExportToolbarComponent exportToolbar() {
        return new ExportToolbarComponent(driver);
    }

    public SidebarComponent sidebar() {
        return new SidebarComponent(driver);
    }
}
