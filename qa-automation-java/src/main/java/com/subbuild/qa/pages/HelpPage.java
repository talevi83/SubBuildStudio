package com.subbuild.qa.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;

import java.util.List;

/**
 * Page Object for the Help & Q&A screen (workflow guide and interactive FAQ accordion).
 */
public class HelpPage extends BasePage {

    private static final By HELP_SCREEN = By.cssSelector("[data-testid='help-qa-screen']");
    private static final By FAQ_SECTION = By.cssSelector("[data-testid='help-qa-section']");
    private static final By FAQ_BUTTONS = By.cssSelector("[data-testid='help-qa-section'] button");
    private static final By BACK_BTN = By.xpath("//*[@data-testid='help-qa-screen']//button[.//*[contains(@class,'lucide-arrow-left')]]");

    public HelpPage(WebDriver driver) {
        super(driver);
    }

    public HelpPage waitForLoaded() {
        waitVisible(HELP_SCREEN);
        waitVisible(FAQ_SECTION);
        return this;
    }

    public int getFaqCount() {
        return waitAllVisible(FAQ_BUTTONS).size();
    }

    public HelpPage toggleFaqItem(int zeroBasedIndex) {
        List<WebElement> buttons = waitAllVisible(FAQ_BUTTONS);
        WebElement btn = buttons.get(zeroBasedIndex);
        scrollIntoView(btn);
        btn.click();
        return this;
    }

    public IngestPage backToStudio() {
        click(BACK_BTN);
        return new IngestPage(driver);
    }
}
