package com.subbuild.qa.config;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.Locale;

/**
 * Central runtime configuration for the Java QA automation suite.
 * Supports overrides via JVM system properties (-Dsubbuild.*) or environment variables.
 */
public final class TestConfig {

    private TestConfig() {
    }

    public static String browser() {
        return read("subbuild.browser", "SUBBUILD_QA_BROWSER", "chrome").toLowerCase(Locale.ROOT);
    }

    public static boolean headless() {
        String explicit = read("subbuild.headless", "SUBBUILD_QA_HEADLESS", "");
        if (!explicit.isBlank()) {
            return !"false".equalsIgnoreCase(explicit);
        }
        return isCiEnvironment() || (isLinux() && isBlankEnv("DISPLAY") && isBlankEnv("WAYLAND_DISPLAY"));
    }

    public static Duration defaultTimeout() {
        long seconds = Long.parseLong(read("subbuild.timeoutSeconds", "SUBBUILD_QA_TIMEOUT", "15"));
        return Duration.ofSeconds(seconds);
    }

    public static String explicitBaseUrl() {
        return read("subbuild.baseUrl", "SUBBUILD_QA_BASE_URL", "");
    }

    public static String explicitToken() {
        return read("subbuild.token", "SUBBUILD_TOKEN", "");
    }

    public static String chromeBinary() {
        String explicit = read("subbuild.chromeBinary", "SUBBUILD_QA_CHROME_BINARY", "");
        if (!explicit.isBlank()) {
            return explicit;
        }
        String envBin = System.getenv("CHROME_BIN");
        if (envBin != null && !envBin.isBlank()) {
            return envBin.trim();
        }
        if (isLinux()) {
            // Preserve standard Chrome/Selenium Manager resolution on x86_64 CI (e.g. GitHub Actions).
            if (!isArm() && Files.isExecutable(Paths.get("/usr/bin/google-chrome"))) {
                return "";
            }
            Path found = firstExecutable(
                    "/usr/bin/chromium",
                    "/usr/bin/chromium-browser",
                    "/snap/bin/chromium",
                    "/usr/bin/google-chrome"
            );
            if (found != null) {
                return found.toString();
            }
        }
        return "";
    }

    public static String chromeDriverPath() {
        String sysProp = System.getProperty("webdriver.chrome.driver");
        if (sysProp != null && !sysProp.isBlank() && !sysProp.startsWith("${")) {
            return sysProp.trim();
        }
        String explicit = read("subbuild.chromeDriver", "SUBBUILD_QA_CHROME_DRIVER", "");
        if (!explicit.isBlank()) {
            return explicit;
        }
        String envDriver = System.getenv("CHROMEDRIVER_BIN");
        if (envDriver != null && !envDriver.isBlank()) {
            return envDriver.trim();
        }
        // Google Chrome for Testing has no Linux ARM64 builds (e.g. Raspberry Pi),
        // so resolve the OS-packaged chromedriver when running on ARM or Chromium-only Linux.
        if (isLinux() && (isArm() || !Files.isExecutable(Paths.get("/usr/bin/google-chrome")))) {
            Path found = firstExecutable(
                    "/usr/bin/chromedriver",
                    "/usr/lib/chromium/chromedriver",
                    "/usr/lib/chromium-browser/chromedriver"
            );
            if (found != null) {
                return found.toString();
            }
        }
        return "";
    }

    public static Path projectRoot() {
        Path cwd = Paths.get("").toAbsolutePath().normalize();
        if (Files.isDirectory(cwd.resolve("backend"))) {
            return cwd;
        }
        if (cwd.getFileName() != null && "qa-automation-java".equals(cwd.getFileName().toString())) {
            return cwd.getParent();
        }
        return cwd;
    }

    public static Path backendDir() {
        return projectRoot().resolve("backend").normalize();
    }

    public static Path pythonExecutable() {
        String explicit = read("subbuild.python", "SUBBUILD_QA_PYTHON", "");
        if (!explicit.isBlank()) {
            return Paths.get(explicit);
        }
        Path winVenv = backendDir().resolve(".venv").resolve("Scripts").resolve("python.exe");
        if (Files.isExecutable(winVenv)) {
            return winVenv;
        }
        Path unixVenv3 = backendDir().resolve(".venv").resolve("bin").resolve("python3");
        if (Files.isExecutable(unixVenv3)) {
            return unixVenv3;
        }
        Path unixVenv = backendDir().resolve(".venv").resolve("bin").resolve("python");
        if (Files.isExecutable(unixVenv)) {
            return unixVenv;
        }
        if (!isWindows()) {
            Path onPath = findExecutableOnPath("python3", "python");
            if (onPath != null) {
                return onPath;
            }
            return Paths.get("python3");
        }
        return Paths.get("python");
    }

    public static Path reportDir() {
        return Paths.get("target", "reports").toAbsolutePath().normalize();
    }

    private static boolean isWindows() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("win");
    }

    private static boolean isLinux() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("linux");
    }

    private static boolean isArm() {
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        return arch.contains("aarch64") || arch.contains("arm");
    }

    private static boolean isCiEnvironment() {
        return !isBlankEnv("CI") || !isBlankEnv("JENKINS_URL") || !isBlankEnv("GITHUB_ACTIONS");
    }

    private static boolean isBlankEnv(String key) {
        String val = System.getenv(key);
        return val == null || val.isBlank();
    }

    private static Path firstExecutable(String... candidates) {
        for (String candidate : candidates) {
            Path path = Paths.get(candidate);
            if (Files.isExecutable(path)) {
                return path;
            }
        }
        return null;
    }

    private static Path findExecutableOnPath(String... names) {
        String pathEnv = System.getenv("PATH");
        if (pathEnv == null || pathEnv.isBlank()) {
            return null;
        }
        String sep = java.io.File.pathSeparator;
        for (String dir : pathEnv.split(java.util.regex.Pattern.quote(sep))) {
            if (dir.isBlank()) {
                continue;
            }
            Path base = Paths.get(dir);
            for (String name : names) {
                Path candidate = base.resolve(name);
                if (Files.isExecutable(candidate) && !Files.isDirectory(candidate)) {
                    return candidate;
                }
            }
        }
        return null;
    }

    private static String read(String propKey, String envKey, String fallback) {
        String prop = System.getProperty(propKey);
        if (prop != null && !prop.isBlank() && !prop.startsWith("${")) {
            return prop.trim();
        }
        String env = System.getenv(envKey);
        if (env != null && !env.isBlank()) {
            return env.trim();
        }
        return fallback;
    }
}
