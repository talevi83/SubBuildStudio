package com.subbuild.qa.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;

import java.util.List;

/**
 * Left navigation sidebar: New Project, job list, bulk deletion, Settings, and Help.
 */
public class SidebarComponent extends BasePage {

    private static final By NEW_PROJECT_BTN = By.xpath("//aside//button[contains(@class,'bg-indigo-600')]");
    private static final By SETTINGS_BTN = By.xpath("//aside//button[.//*[contains(@class,'lucide-settings')]]");
    private static final By HELP_BTN = By.cssSelector("[data-testid='sidebar-help-btn']");
    private static final By CLEAR_ALL_BTN = By.cssSelector("[data-testid='sidebar-clear-all-btn']");
    private static final By SELECT_ALL_CHECKBOX = By.cssSelector("[data-testid='sidebar-select-all']");
    private static final By DELETE_SELECTED_BTN = By.cssSelector("[data-testid='sidebar-delete-selected-btn']");
    private static final By EMPTY_STATE = By.xpath("//aside//div[contains(@class,'overflow-y-auto')]//div[contains(@class,'text-center')]");

    public SidebarComponent(WebDriver driver) {
        super(driver);
    }

    public void clickNewProject() {
        click(NEW_PROJECT_BTN);
    }

    public SettingsPage openSettings() {
        click(SETTINGS_BTN);
        return new SettingsPage(driver).waitForLoaded();
    }

    public HelpPage openHelp() {
        click(HELP_BTN);
        return new HelpPage(driver).waitForLoaded();
    }

    public void selectJobById(String jobId) {
        By jobCheckbox = By.cssSelector("[data-testid='sidebar-select-" + jobId + "']");
        WebElement checkbox = waitVisible(jobCheckbox);
        WebElement card = checkbox.findElement(By.xpath("./ancestor::div[contains(@class,'group')][1]"));
        scrollIntoView(card);
        card.click();
    }

    public boolean hasJobCard(String jobId) {
        return !driver.findElements(By.cssSelector("[data-testid='sidebar-select-" + jobId + "']")).isEmpty();
    }

    public void waitForJobCard(String jobId) {
        waitVisible(By.cssSelector("[data-testid='sidebar-select-" + jobId + "']"));
    }

    public void deleteAllViaClearAll() {
        click(CLEAR_ALL_BTN);
        acceptConfirmDialog();
        wait.until(ExpectedConditions.visibilityOfElementLocated(EMPTY_STATE));
    }

    public void selectAllAndDeleteSelected() {
        click(SELECT_ALL_CHECKBOX);
        click(DELETE_SELECTED_BTN);
        acceptConfirmDialog();
        wait.until(ExpectedConditions.visibilityOfElementLocated(EMPTY_STATE));
    }

    public int jobCount() {
        List<WebElement> items = driver.findElements(By.cssSelector("aside [data-testid^='sidebar-select-']:not([data-testid='sidebar-select-all'])"));
        return items.size();
    }
}
