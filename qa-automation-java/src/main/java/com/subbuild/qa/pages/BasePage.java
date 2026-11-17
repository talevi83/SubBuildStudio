package com.subbuild.qa.pages;

import com.subbuild.qa.config.TestConfig;
import org.openqa.selenium.Alert;
import org.openqa.selenium.By;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.Keys;
import org.openqa.selenium.NoSuchElementException;
import org.openqa.selenium.StaleElementReferenceException;
import org.openqa.selenium.TimeoutException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.Select;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;
import java.util.List;

/**
 * Base Page Object providing explicit WebDriverWait primitives and resilient DOM interactions.
 * Never uses Thread.sleep.
 */
public abstract class BasePage {

    protected final WebDriver driver;
    protected final WebDriverWait wait;

    protected BasePage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, TestConfig.defaultTimeout());
        this.wait.ignoring(StaleElementReferenceException.class);
    }

    protected WebElement waitVisible(By locator) {
        return wait.until(ExpectedConditions.visibilityOfElementLocated(locator));
    }

    protected List<WebElement> waitAllVisible(By locator) {
        return wait.until(ExpectedConditions.visibilityOfAllElementsLocatedBy(locator));
    }

    protected WebElement waitClickable(By locator) {
        return wait.until(ExpectedConditions.elementToBeClickable(locator));
    }

    protected void waitInvisible(By locator) {
        wait.until(ExpectedConditions.invisibilityOfElementLocated(locator));
    }

    protected void click(By locator) {
        waitClickable(locator).click();
    }

    protected void clearAndType(By locator, String text) {
        WebElement el = waitVisible(locator);
        el.click();
        el.sendKeys(Keys.chord(Keys.CONTROL, "a"));
        el.sendKeys(Keys.DELETE);
        if (text != null && !text.isEmpty()) {
            el.sendKeys(text);
        }
    }

    protected void selectByValue(By locator, String value) {
        WebElement el = waitVisible(locator);
        new Select(el).selectByValue(value);
    }

    protected String selectedValue(By locator) {
        WebElement el = waitVisible(locator);
        return new Select(el).getFirstSelectedOption().getAttribute("value");
    }

    protected boolean isDisplayedQuick(By locator, Duration timeout) {
        try {
            new WebDriverWait(driver, timeout)
                    .ignoring(NoSuchElementException.class, StaleElementReferenceException.class)
                    .until(ExpectedConditions.visibilityOfElementLocated(locator));
            return true;
        } catch (TimeoutException e) {
            return false;
        }
    }

    protected void scrollIntoView(WebElement element) {
        ((JavascriptExecutor) driver).executeScript(
                "arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", element);
    }

    protected void acceptConfirmDialog() {
        Alert alert = wait.until(ExpectedConditions.alertIsPresent());
        alert.accept();
    }

    public String getDocumentDir() {
        return (String) ((JavascriptExecutor) driver).executeScript("return document.documentElement.dir;");
    }

    public String getDocumentLang() {
        return (String) ((JavascriptExecutor) driver).executeScript("return document.documentElement.lang;");
    }

    public String getDocumentTheme() {
        return (String) ((JavascriptExecutor) driver).executeScript("return document.documentElement.dataset.theme;");
    }
}
