package com.subbuild.qa.driver;

import com.subbuild.qa.config.TestConfig;
import org.openqa.selenium.Dimension;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeDriverService;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.edge.EdgeDriver;
import org.openqa.selenium.edge.EdgeOptions;

import java.nio.file.Path;
import java.time.Duration;

/**
 * Factory creating Selenium WebDriver instances with explicit-wait-only discipline
 * (implicit wait strictly set to ZERO).
 */
public final class DriverFactory {

    private DriverFactory() {
    }

    public static WebDriver createDriver() {
        String browser = TestConfig.browser();
        boolean headless = TestConfig.headless();

        WebDriver driver = switch (browser) {
            case "edge" -> createEdge(headless);
            case "chrome" -> createChrome(headless);
            default -> throw new IllegalArgumentException("Unsupported browser: " + browser);
        };

        driver.manage().timeouts().implicitlyWait(Duration.ZERO);
        driver.manage().timeouts().pageLoadTimeout(Duration.ofSeconds(30));
        driver.manage().window().setSize(new Dimension(1600, 1000));
        return driver;
    }

    private static WebDriver createChrome(boolean headless) {
        ChromeOptions options = new ChromeOptions();
        String chromeBinary = TestConfig.chromeBinary();
        if (!chromeBinary.isBlank()) {
            options.setBinary(chromeBinary);
        }
        if (headless) {
            options.addArguments("--headless=new");
        }
        options.addArguments(
                "--window-size=1600,1000",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--remote-allow-origins=*"
        );
        String driverPath = TestConfig.chromeDriverPath();
        if (!driverPath.isBlank()) {
            ChromeDriverService service = new ChromeDriverService.Builder()
                    .usingDriverExecutable(Path.of(driverPath).toFile())
                    .build();
            return new ChromeDriver(service, options);
        }
        return new ChromeDriver(options);
    }

    private static WebDriver createEdge(boolean headless) {
        EdgeOptions options = new EdgeOptions();
        if (headless) {
            options.addArguments("--headless=new");
        }
        options.addArguments(
                "--window-size=1600,1000",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage"
        );
        return new EdgeDriver(options);
    }
}
