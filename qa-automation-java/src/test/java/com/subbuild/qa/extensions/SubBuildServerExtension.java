package com.subbuild.qa.extensions;

import com.subbuild.qa.config.TestConfig;
import org.junit.jupiter.api.extension.BeforeAllCallback;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.api.extension.ParameterContext;
import org.junit.jupiter.api.extension.ParameterResolver;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.locks.LockSupport;
import java.util.stream.Stream;

/**
 * Starts an isolated SubBuild Studio backend server once per test session using a temporary
 * storage directory and SQLite DB so real user data in backend/storage and backend/.env is
 * never modified.
 */
public class SubBuildServerExtension implements BeforeAllCallback, ParameterResolver {

    private static final ExtensionContext.Namespace NS =
            ExtensionContext.Namespace.create(SubBuildServerExtension.class);
    private static final String RESOURCE_KEY = "subbuild-server";

    @Override
    public void beforeAll(ExtensionContext context) throws Exception {
        ensureStarted(context);
    }

    @Override
    public boolean supportsParameter(ParameterContext parameterContext, ExtensionContext extensionContext) {
        return parameterContext.getParameter().getType().equals(ServerHandle.class);
    }

    @Override
    public Object resolveParameter(ParameterContext parameterContext, ExtensionContext extensionContext) {
        return ensureStarted(extensionContext);
    }

    private static ServerHandle ensureStarted(ExtensionContext context) {
        ExtensionContext.Store rootStore = context.getRoot().getStore(NS);
        return rootStore.getOrComputeIfAbsent(RESOURCE_KEY, key -> ServerHandle.startOrAttach(), ServerHandle.class);
    }

    public static ServerHandle getHandle(ExtensionContext context) {
        return context.getRoot().getStore(NS).get(RESOURCE_KEY, ServerHandle.class);
    }

    public static final class ServerHandle implements ExtensionContext.Store.CloseableResource {

        private final String baseUrl;
        private final String token;
        private final Path tempStorageDir;
        private final Path sampleVideoPath;
        private final Path sampleSrtPath;
        private final Process process;

        private ServerHandle(
                String baseUrl,
                String token,
                Path tempStorageDir,
                Path sampleVideoPath,
                Path sampleSrtPath,
                Process process
        ) {
            this.baseUrl = baseUrl;
            this.token = token;
            this.tempStorageDir = tempStorageDir;
            this.sampleVideoPath = sampleVideoPath;
            this.sampleSrtPath = sampleSrtPath;
            this.process = process;
        }

        public String baseUrl() {
            return baseUrl;
        }

        public String token() {
            return token;
        }

        public Path sampleVideoPath() {
            return sampleVideoPath;
        }

        public Path sampleSrtPath() {
            return sampleSrtPath;
        }

        static ServerHandle startOrAttach() {
            try {
                Path tempDir = Files.createTempDirectory("subbuild_qa_java_");
                Path srtPath = writeSampleSrt(tempDir.resolve("sample.srt"));
                Path videoPath = generateSampleMkv(srtPath, tempDir.resolve("qa_sample.mkv"));

                String explicitUrl = TestConfig.explicitBaseUrl();
                if (!explicitUrl.isBlank()) {
                    String tok = TestConfig.explicitToken();
                    return new ServerHandle(explicitUrl, tok, tempDir, videoPath, srtPath, null);
                }

                int port = findFreePort();
                String token = "qa-java-token-" + UUID.randomUUID().toString().replace("-", "");
                String baseUrl = "http://127.0.0.1:" + port;

                Path reportDir = TestConfig.reportDir();
                Files.createDirectories(reportDir);
                Path serverLog = reportDir.resolve("server.log");

                ProcessBuilder pb = new ProcessBuilder(
                        TestConfig.pythonExecutable().toString(),
                        "-m", "uvicorn", "app.main:app",
                        "--host", "127.0.0.1",
                        "--port", String.valueOf(port)
                );
                pb.directory(TestConfig.backendDir().toFile());
                pb.redirectErrorStream(true);
                pb.redirectOutput(serverLog.toFile());

                var env = pb.environment();
                env.put("STORAGE_DIR", tempDir.toAbsolutePath().toString());
                env.put("DATABASE_URL", "sqlite:///" + tempDir.resolve("qa_test.db").toString().replace('\\', '/'));
                env.put("SUBBUILD_TOKEN", token);
                env.put("GEMINI_API_KEY", "");
                env.put("ALLOWED_HOSTS", "[\"localhost\",\"127.0.0.1\"]");

                Process proc = pb.start();
                waitUntilReady(baseUrl, token, proc, serverLog);
                return new ServerHandle(baseUrl, token, tempDir, videoPath, srtPath, proc);
            } catch (Exception e) {
                throw new IllegalStateException("Failed to bootstrap SubBuild Studio QA server", e);
            }
        }

        private static int findFreePort() throws IOException {
            try (ServerSocket socket = new ServerSocket(0)) {
                return socket.getLocalPort();
            }
        }

        private static void waitUntilReady(String baseUrl, String token, Process proc, Path serverLog)
                throws IOException {
            HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).build();
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/system/info"))
                    .header("X-SubBuild-Token", token)
                    .timeout(Duration.ofSeconds(2))
                    .GET()
                    .build();

            Instant deadline = Instant.now().plus(Duration.ofSeconds(30));
            while (Instant.now().isBefore(deadline)) {
                if (!proc.isAlive()) {
                    String log = Files.exists(serverLog) ? Files.readString(serverLog) : "";
                    throw new IllegalStateException("Backend server exited prematurely (" + proc.exitValue() + "):\n" + log);
                }
                try {
                    HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
                    if (resp.statusCode() == 200) {
                        return;
                    }
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("Interrupted while waiting for server", ie);
                } catch (Exception ignored) {
                    // Server still starting up
                }
                LockSupport.parkNanos(150_000_000L);
            }
            String log = Files.exists(serverLog) ? Files.readString(serverLog) : "";
            throw new IllegalStateException("Timed out waiting for SubBuild backend at " + baseUrl + "\n" + log);
        }

        private static Path writeSampleSrt(Path target) throws IOException {
            String srt = """
                    1
                    00:00:00,300 --> 00:00:01,500
                    Hello from SubBuild QA.

                    2
                    00:00:01,700 --> 00:00:02,800
                    Testing automated subtitle workflows.

                    3
                    00:00:03,000 --> 00:00:03,800
                    Everything is synced cleanly.
                    """;
            Files.writeString(target, srt, StandardCharsets.UTF_8);
            return target;
        }

        private static Path generateSampleMkv(Path srtPath, Path outputMkv) throws IOException, InterruptedException {
            ProcessBuilder pb = new ProcessBuilder(
                    "ffmpeg", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=160x90:d=4",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                    "-i", srtPath.toAbsolutePath().toString(),
                    "-map", "0:v", "-map", "1:a", "-map", "2:s",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-c:s", "srt",
                    outputMkv.toAbsolutePath().toString()
            );
            pb.redirectErrorStream(true);
            Process p = pb.start();
            String output = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            int code = p.waitFor();
            if (code != 0) {
                throw new IllegalStateException("ffmpeg failed to generate sample fixture (exit " + code + "): " + output);
            }
            return outputMkv;
        }

        @Override
        public void close() {
            if (process != null) {
                try {
                    List<ProcessHandle> children = process.descendants().toList();
                    children.forEach(ProcessHandle::destroyForcibly);
                    process.destroyForcibly();
                    process.waitFor();
                } catch (Exception ignored) {
                }
            }
            if (tempStorageDir != null && Files.exists(tempStorageDir)) {
                try (Stream<Path> walk = Files.walk(tempStorageDir)) {
                    walk.sorted(Comparator.reverseOrder()).forEach(p -> {
                        try {
                            Files.deleteIfExists(p);
                        } catch (IOException ignored) {
                        }
                    });
                } catch (IOException ignored) {
                }
            }
        }
    }
}
