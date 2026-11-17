package com.subbuild.qa.base;

import com.subbuild.qa.driver.DriverFactory;
import com.subbuild.qa.pages.IngestPage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.openqa.selenium.WebDriver;

/**
 * Base class for Selenium WebDriver UI tests.
 * Manages WebDriver lifecycle per test, ensures clean studio state, and exposes the driver
 * so ExtentReportExtension can capture screenshots automatically on failure.
 */
public abstract class BaseTest extends BaseApiTest {

    protected WebDriver driver;

    @BeforeEach
    void initDriver() {
        driver = DriverFactory.createDriver();
    }

    @AfterEach
    void quitDriver() {
        if (driver != null) {
            try {
                driver.quit();
            } finally {
                driver = null;
            }
        }
    }

    public WebDriver getDriverIfInitialized() {
        return driver;
    }

    /**
     * Authenticates the browser via the one-time /?token=... bootstrap URL and dismisses
     * the first-run Gemini API key modal when it appears.
     */
    protected IngestPage openStudio() {
        return new IngestPage(driver)
                .openWithToken(baseUrl, sessionToken)
                .dismissFirstRunModalIfPresent();
    }

    /**
     * Authenticates the browser via /?token=... without dismissing the first-run modal.
     */
    protected IngestPage openStudioRaw() {
        return new IngestPage(driver).openWithToken(baseUrl, sessionToken);
    }
}
