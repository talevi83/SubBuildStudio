package com.subbuild.qa.extensions;

import com.aventstack.extentreports.ExtentReports;
import com.aventstack.extentreports.ExtentTest;
import com.aventstack.extentreports.MediaEntityBuilder;
import com.aventstack.extentreports.reporter.ExtentSparkReporter;
import com.aventstack.extentreports.reporter.configuration.Theme;
import com.subbuild.qa.base.BaseTest;
import com.subbuild.qa.config.TestConfig;
import org.junit.jupiter.api.extension.AfterTestExecutionCallback;
import org.junit.jupiter.api.extension.BeforeAllCallback;
import org.junit.jupiter.api.extension.BeforeEachCallback;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.api.extension.TestWatcher;
import org.openqa.selenium.OutputType;
import org.openqa.selenium.TakesScreenshot;
import org.openqa.selenium.WebDriver;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Optional;

/**
 * Generates an interactive visual HTML dashboard (ExtentReports) at target/reports/subbuild-qa-report.html
 * and automatically captures + embeds browser screenshots whenever a UI test fails (before @AfterEach quits WebDriver).
 */
public class ExtentReportExtension
        implements BeforeAllCallback, BeforeEachCallback, AfterTestExecutionCallback, TestWatcher {

    private static final ExtensionContext.Namespace NS =
            ExtensionContext.Namespace.create(ExtentReportExtension.class);
    private static final String REPORT_KEY = "extent-report";
    private static final String TEST_KEY = "current-extent-test";
    private static final String SCREENSHOT_KEY = "failure-screenshot-rel";

    @Override
    public void beforeAll(ExtensionContext context) {
        context.getRoot().getStore(NS).getOrComputeIfAbsent(
                REPORT_KEY, k -> new ReportHolder(), ReportHolder.class);
    }

    @Override
    public void beforeEach(ExtensionContext context) {
        ReportHolder holder = context.getRoot().getStore(NS).get(REPORT_KEY, ReportHolder.class);
        if (holder == null) {
            return;
        }
        String className = context.getRequiredTestClass().getSimpleName();
        String testName = context.getDisplayName();
        ExtentTest extentTest = holder.extent.createTest(className + " :: " + testName);
        context.getTags().forEach(extentTest::assignCategory);
        context.getStore(NS).put(TEST_KEY, extentTest);
    }

    @Override
    public void afterTestExecution(ExtensionContext context) {
        if (context.getExecutionException().isEmpty()) {
            return;
        }
        Optional<Object> instance = context.getTestInstance();
        if (instance.isPresent() && instance.get() instanceof BaseTest uiTest) {
            WebDriver driver = uiTest.getDriverIfInitialized();
            if (driver instanceof TakesScreenshot ts) {
                try {
                    Path screenshotsDir = TestConfig.reportDir().resolve("screenshots");
                    Files.createDirectories(screenshotsDir);
                    String safeName = context.getRequiredTestMethod().getName().replaceAll("[^A-Za-z0-9_-]", "_");
                    String stamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS"));
                    Path target = screenshotsDir.resolve(safeName + "_" + stamp + ".png");
                    Files.write(target, ts.getScreenshotAs(OutputType.BYTES));
                    context.getStore(NS).put(SCREENSHOT_KEY, "screenshots/" + target.getFileName());
                } catch (Exception ignored) {
                }
            }
        }
    }

    @Override
    public void testSuccessful(ExtensionContext context) {
        ExtentTest test = context.getStore(NS).get(TEST_KEY, ExtentTest.class);
        if (test != null) {
            test.pass("Test passed");
        }
    }

    @Override
    public void testAborted(ExtensionContext context, Throwable cause) {
        ExtentTest test = context.getStore(NS).get(TEST_KEY, ExtentTest.class);
        if (test != null) {
            test.skip("Test aborted: " + (cause != null ? cause.getMessage() : "no reason"));
        }
    }

    @Override
    public void testDisabled(ExtensionContext context, Optional<String> reason) {
        ReportHolder holder = context.getRoot().getStore(NS).get(REPORT_KEY, ReportHolder.class);
        if (holder != null) {
            ExtentTest test = holder.extent.createTest(
                    context.getRequiredTestClass().getSimpleName() + " :: " + context.getDisplayName());
            context.getTags().forEach(test::assignCategory);
            test.skip(reason.orElse("Disabled"));
        }
    }

    @Override
    public void testFailed(ExtensionContext context, Throwable cause) {
        ExtentTest test = context.getStore(NS).get(TEST_KEY, ExtentTest.class);
        if (test == null) {
            return;
        }
        String screenshotRel = context.getStore(NS).get(SCREENSHOT_KEY, String.class);
        if (screenshotRel != null) {
            try {
                test.fail(cause, MediaEntityBuilder.createScreenCaptureFromPath(screenshotRel).build());
                return;
            } catch (Exception ignored) {
            }
        }
        test.fail(cause);
    }

    private static final class ReportHolder implements ExtensionContext.Store.CloseableResource {
        private final ExtentReports extent;

        ReportHolder() {
            try {
                Path reportDir = TestConfig.reportDir();
                Files.createDirectories(reportDir);
                Path htmlPath = reportDir.resolve("subbuild-qa-report.html");

                ExtentSparkReporter spark = new ExtentSparkReporter(htmlPath.toFile());
                spark.config().setTheme(Theme.DARK);
                spark.config().setDocumentTitle("SubBuild Studio — QA Automation Report");
                spark.config().setReportName("SubBuild Studio Black-Box E2E & API Suite");

                this.extent = new ExtentReports();
                this.extent.attachReporter(spark);
                this.extent.setSystemInfo("Browser", TestConfig.browser());
                this.extent.setSystemInfo("Headless", String.valueOf(TestConfig.headless()));
                this.extent.setSystemInfo("Java Version", System.getProperty("java.version"));
            } catch (Exception e) {
                throw new IllegalStateException("Could not initialize ExtentReports", e);
            }
        }

        @Override
        public void close() {
            extent.flush();
        }
    }
}
