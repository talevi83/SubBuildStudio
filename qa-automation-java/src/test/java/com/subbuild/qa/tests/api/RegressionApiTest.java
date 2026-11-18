package com.subbuild.qa.tests.api;

import com.subbuild.qa.base.BaseApiTest;
import com.sun.net.httpserver.HttpServer;
import io.restassured.response.Response;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.locks.LockSupport;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("API Regression Suite (Edge Cases, Validation, Split/Merge/Shift, Glossary & Embed)")
@Tag("regression")
@Tag("api")
class RegressionApiTest extends BaseApiTest {

    @AfterEach
    void cleanupJobs() {
        api.deleteAllJobs();
    }

    @Test
    @DisplayName("Split, merge, and shift segments record edit history and clamp negative shifts at 0.0")
    void splitMergeAndShiftSegmentsWithHistory() {
        String jobId = api.seedCompletedEmbeddedJob(sampleVideoPath.toString());
        List<Map<String, Object>> initial = api.getJob(jobId).jsonPath().getList("segments");
        long secondId = ((Number) initial.get(1).get("id")).longValue();

        // Split segment #2 into two segments
        Response split = api.splitSegment(jobId, secondId, null);
        assertEquals(200, split.statusCode(), split.asString());
        List<Map<String, Object>> afterSplit = split.jsonPath().getList("segments");
        assertEquals(4, afterSplit.size());

        // Merge the two split segments back together
        long part1Id = ((Number) afterSplit.get(1).get("id")).longValue();
        long part2Id = ((Number) afterSplit.get(2).get("id")).longValue();
        Response merged = api.mergeSegments(jobId, List.of(part1Id, part2Id));
        assertEquals(200, merged.statusCode(), merged.asString());
        assertEquals(3, merged.jsonPath().getList("segments").size());

        // Shift forward +1.5s, then shift first segment -10.0s (clamped at 0.0)
        Response shiftedFwd = api.shiftSegments(jobId, 1.5, null);
        assertEquals(200, shiftedFwd.statusCode());
        double shiftedStart = shiftedFwd.jsonPath().getDouble("segments[0].start_time");
        assertTrue(shiftedStart >= 1.79, "Expected first cue shifted forward by +1.5s");

        long firstId = shiftedFwd.jsonPath().getLong("segments[0].id");
        Response shiftedBack = api.shiftSegments(jobId, -10.0, List.of(firstId));
        assertEquals(200, shiftedBack.statusCode());
        assertEquals(0.0, shiftedBack.jsonPath().getDouble("segments[0].start_time"), 0.001);

        // Verify history actions recorded
        List<String> actions = api.getHistory(jobId).jsonPath().getList("entries.action");
        assertTrue(actions.containsAll(List.of("original", "split", "merge", "shift_time")),
                "Missing expected history actions: " + actions);
    }

    @Test
    @DisplayName("Glossary PUT persists character metadata and registers Series Glossary Profile")
    void glossaryUpdateAndSeriesProfilePersistence() {
        String jobId = api.seedCompletedEmbeddedJob(sampleVideoPath.toString());
        String profileName = "QA_Series_" + UUID.randomUUID().toString().substring(0, 6);

        Response updated = api.updateGlossary(jobId, Map.of(
                "summary", "Naval thriller",
                "characters", List.of(Map.of(
                        "name", "Dr. Rachel Scott",
                        "target_name", "ד\"ר רייצ'ל סקוט",
                        "gender", "female",
                        "notes", "Lead virologist"
                )),
                "terms", List.of(Map.of(
                        "source", "Bridge",
                        "translation", "גשר הפיקוד"
                )),
                "register_notes", "Formal military tone",
                "profile", profileName
        ));
        assertEquals(200, updated.statusCode());
        assertEquals("Dr. Rachel Scott", updated.jsonPath().getString("characters[0].name"));

        Response job = api.getJob(jobId);
        assertEquals(profileName, job.jsonPath().getString("glossary_profile"));
        assertEquals("female", job.jsonPath().getString("glossary.characters[0].gender"));

        List<String> profiles = api.listGlossaryProfiles().jsonPath().getList("$");
        assertTrue(profiles.contains(profileName), "Profile should appear in /api/glossary-profiles");
    }

    @Test
    @DisplayName("Bilingual RTL SRT export includes UTF-8 BOM, RLM marks, dual filename, and stripped DB storage")
    void bilingualAndRtlExportContract() {
        String jobId = api.seedCompletedEmbeddedJob(sampleVideoPath.toString());
        long firstId = api.getJob(jobId).jsonPath().getLong("segments[0].id");

        // Patch translated Hebrew text containing explicit RLM — DB stores it clean
        Response patched = api.patchSegment(jobId, firstId, Map.of("translated_text", "\u200fשלום מ-SubBuild QA?"));
        assertEquals(200, patched.statusCode());
        assertFalse(patched.jsonPath().getString("translated_text").contains("\u200f"),
                "Segment text in DB must strip directional marks");

        Response srt = api.exportSrt(jobId, true, true, true);
        assertEquals(200, srt.statusCode());
        assertTrue(srt.getHeader("Content-Disposition").contains(".dual.srt"));
        String body = srt.asString();
        assertTrue(body.contains("Hello from SubBuild QA."));
        assertTrue(body.contains("\u200f"));
        assertTrue(body.contains("שלום מ-SubBuild QA?"));
    }

    @ParameterizedTest(name = "Invalid segment payload rejected ({1}): {0}")
    @CsvSource(delimiter = '|', value = {
            "{\"start_time\": 0.5, \"end_time\": 1.2, \"original_text\": \"overlap\"} | 409",
            "{\"start_time\": 5.0, \"end_time\": 5.0, \"original_text\": \"zero\"} | 422",
            "{\"start_time\": 6.0, \"end_time\": 5.0, \"original_text\": \"reversed\"} | 422",
            "{\"start_time\": -1.0, \"end_time\": 0.5, \"original_text\": \"neg\"} | 422",
            "{\"start_time\": 5.0, \"end_time\": 6.0, \"original_text\": \"   \"} | 422",
            "{\"start_time\": 5.0, \"end_time\": 6.0, \"original_text\": \"bad\", \"segment_type\": \"INVALID\"} | 422"
    })
    @DisplayName("Segment creation enforces overlap (409) and schema/bounds validation (422)")
    void negativeSegmentValidationAndOverlaps(String rawJson, int expectedStatus) {
        String jobId = api.seedCompletedEmbeddedJob(sampleVideoPath.toString());
        Response res = io.restassured.RestAssured.given()
                .baseUri(baseUrl)
                .header("X-SubBuild-Token", sessionToken)
                .contentType(io.restassured.http.ContentType.JSON)
                .body(rawJson)
                .post("/api/jobs/{jobId}/segments", jobId);
        assertEquals(expectedStatus, res.statusCode(), res.asString());
    }

    @Test
    @DisplayName("Invalid settings, thinking level, and missing job IDs return deterministic 400/404 errors")
    void negativeJobAndSettingsErrors() {
        String jobId = api.seedCompletedEmbeddedJob(sampleVideoPath.toString());

        assertEquals(404, api.getJob("job_does_not_exist").statusCode());
        assertEquals(404, api.exportJobLog("job_does_not_exist").statusCode());
        assertEquals(400, api.updateJobSettings(jobId, Map.of("onscreen_position", "invalid_corner")).statusCode());
        assertEquals(400, api.retranslateJob(jobId, Map.of("thinking", "ludicrous")).statusCode());
    }

    @ParameterizedTest(name = "Non-media file extension rejected: {0}")
    @ValueSource(strings = {"secret.env", "notes.txt", "subbuild.db", "script.py"})
    @Tag("security")
    @DisplayName("Probe and job creation reject non-media extensions (400)")
    void negativeNonMediaPathsRejected(String filename, @TempDir Path tmp) throws Exception {
        Path badFile = tmp.resolve(filename);
        Files.writeString(badFile, "sensitive", StandardCharsets.UTF_8);

        assertEquals(400, api.probePath(badFile.toString()).statusCode());
        assertEquals(400, api.createJob(Map.of(
                "video_path", badFile.toString(),
                "enable_translation", false
        )).statusCode());
    }

    @Test
    @DisplayName("Per-job log export and FFmpeg soft-subtitle video embedding succeed end-to-end")
    void jobLogExportAndSoftEmbed(@TempDir Path tmp) throws Exception {
        Path copyVideo = tmp.resolve("embed_test.mkv");
        Files.copy(sampleVideoPath, copyVideo);

        String jobId = api.seedCompletedEmbeddedJob(copyVideo.toString());

        // Job log export
        Response logRes = api.exportJobLog(jobId);
        assertEquals(200, logRes.statusCode());
        String logText = logRes.asString();
        assertTrue(logText.contains("Job:        " + jobId));
        assertTrue(logText.contains("Status:     COMPLETED"));

        // Soft subtitle embedding into MKV via real ffmpeg
        Response embedStart = api.startEmbedVideo(jobId, Map.of(
                "mode", "soft",
                "use_translated", false
        ));
        assertEquals(200, embedStart.statusCode(), embedStart.asString());
        String taskId = embedStart.jsonPath().getString("task_id");

        Instant deadline = Instant.now().plus(Duration.ofSeconds(20));
        Response statusRes = embedStart;
        while (Instant.now().isBefore(deadline)) {
            statusRes = api.getEmbedStatus(jobId, taskId);
            if (!"running".equals(statusRes.jsonPath().getString("state"))) {
                break;
            }
            LockSupport.parkNanos(150_000_000L);
        }
        assertEquals("done", statusRes.jsonPath().getString("state"), statusRes.asString());
        Path outFile = Path.of(statusRes.jsonPath().getString("path"));
        assertTrue(Files.isRegularFile(outFile), "Embedded MKV output file should exist on disk");
    }

    @Test
    @DisplayName("Multi-provider API key vault stores multiple masked keys per provider and rejects invalid URLs")
    void multiProviderKeyVaultMasksSecretsAndManagesDefaults() {
        // Negative validation checks
        assertEquals(400, api.addProviderKey(Map.of("provider", "openai", "api_key", "")).statusCode());
        assertEquals(400, api.addProviderKey(Map.of("provider", "ollama", "base_url", "ftp://invalid")).statusCode());
        assertEquals(400, api.addProviderKey(Map.of(
                "provider", "ollama", "base_url", "https://ollama.com", "api_key", ""
        )).statusCode());

        // Add 2 keys for OpenAI + 1 for Anthropic + 1 Ollama Cloud
        Response oai1 = api.addProviderKey(Map.of(
                "provider", "openai", "label", "Personal", "api_key", "sk-oai-secret-1111aaaa", "verify", false
        ));
        assertEquals(200, oai1.statusCode());
        String id1 = oai1.jsonPath().getString("key.id");

        Response oai2 = api.addProviderKey(Map.of(
                "provider", "openai", "label", "Work", "api_key", "sk-oai-secret-2222bbbb", "is_default", true, "verify", false
        ));
        assertEquals(200, oai2.statusCode());
        String id2 = oai2.jsonPath().getString("key.id");

        Response ant = api.addProviderKey(Map.of(
                "provider", "anthropic", "label", "Claude Main", "api_key", "sk-ant-secret-9999zzzz", "verify", false
        ));
        assertEquals(200, ant.statusCode());
        String antId = ant.jsonPath().getString("key.id");

        Response keys = api.listProviderKeys();
        assertEquals(200, keys.statusCode());
        String rawJson = keys.asString();
        assertFalse(rawJson.contains("sk-oai-secret-1111aaaa"), "Raw OpenAI key must never be returned");
        assertFalse(rawJson.contains("sk-oai-secret-2222bbbb"), "Raw OpenAI key must never be returned");
        assertFalse(rawJson.contains("sk-ant-secret-9999zzzz"), "Raw Anthropic key must never be returned");
        assertEquals(2, keys.jsonPath().getInt("providers.openai.key_count"));
        assertEquals(id2, keys.jsonPath().getString("providers.openai.default_key_id"));
        assertTrue(keys.jsonPath().getBoolean("providers.anthropic.configured"));

        // Deleting default OpenAI key promotes remaining key automatically
        assertEquals(200, api.deleteProviderKey(id2).statusCode());
        Response afterDel = api.listProviderKeys();
        assertEquals(1, afterDel.jsonPath().getInt("providers.openai.key_count"));
        assertEquals(id1, afterDel.jsonPath().getString("providers.openai.default_key_id"));

        // Cleanup remaining test keys
        api.deleteProviderKey(id1);
        api.deleteProviderKey(antId);
    }

    @Test
    @DisplayName("Mixed-provider settings (OpenAI + Anthropic + Ollama) persist and calculate per-stage cost estimates")
    void mixedProviderSettingsAndCostEstimateAcrossProviders() {
        Map<String, Object> original = api.getSettings().jsonPath().getMap("settings");
        try {
            Map<String, Object> mixed = new java.util.HashMap<>(original);
            mixed.put("translation", Map.of("provider", "openai", "model", "gpt-5-mini", "thinking", "default", "flex", false));
            mixed.put("glossary", Map.of("provider", "anthropic", "model", "claude-haiku-4-5", "thinking", "default", "flex", false));
            mixed.put("vision", Map.of("provider", "ollama", "model", "qwen2.5vl:7b", "thinking", "default", "flex", false));

            Response put = api.updateSettings(mixed);
            assertEquals(200, put.statusCode(), put.asString());
            assertEquals("openai", put.jsonPath().getString("settings.translation.provider"));
            assertEquals("anthropic", put.jsonPath().getString("settings.glossary.provider"));
            assertEquals("ollama", put.jsonPath().getString("settings.vision.provider"));

            Response est = api.estimateCost(Map.of(
                    "duration_seconds", 2700.0,
                    "enable_translation", true,
                    "detect_onscreen_text", true
            ));
            assertEquals(200, est.statusCode());
            assertEquals("openai", est.jsonPath().getString("stages.translation.provider"));
            assertEquals("anthropic", est.jsonPath().getString("stages.glossary.provider"));
            assertEquals("ollama", est.jsonPath().getString("stages.onscreen.provider"));
            assertEquals(0.0, est.jsonPath().getDouble("stages.onscreen.cost_usd"), 0.0001);
            assertTrue(est.jsonPath().getDouble("total_cost_usd") > 0.0);
        } finally {
            api.updateSettings(original);
        }
    }

    @Test
    @DisplayName("Settings catalog exposes episode/movie benchmarks, strict step capabilities, and key-scoped live filtering")
    void catalogExposesEpisodeAndMovieBenchmarksAndStepCapabilities() throws IOException {
        Response res = api.getSettings();
        assertEquals(200, res.statusCode());
        List<Map<String, Object>> catalog = res.jsonPath().getList("catalog");
        assertTrue(catalog.size() >= 10, "Catalog should include multi-provider models");

        Map<String, Object> mini = catalog.stream()
                .filter(m -> "openai".equals(m.get("provider")) && "gpt-5-mini".equals(m.get("id")))
                .findFirst()
                .orElseThrow();
        assertTrue(((Number) mini.get("episode_cost_usd")).doubleValue() > 0.0);
        assertTrue(((Number) mini.get("movie_cost_usd")).doubleValue()
                > ((Number) mini.get("episode_cost_usd")).doubleValue());
        assertEquals(List.of("translation", "glossary", "vision"), mini.get("tasks"));

        Map<String, Object> nano = catalog.stream()
                .filter(m -> "openai".equals(m.get("provider")) && "gpt-5-nano".equals(m.get("id")))
                .findFirst()
                .orElseThrow();
        assertEquals(List.of("vision"), nano.get("tasks"));

        // Verify live key-scoped discovery removes unverified seeds and filters step tasks per model
        HttpServer mockNode = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        mockNode.createContext("/api/tags", exchange -> {
            byte[] body = "{\"models\":[{\"name\":\"qwen2.5:14b\"},{\"name\":\"qwen2.5:1.5b\"},{\"name\":\"qwen2.5-coder:7b\"},{\"name\":\"nomic-embed-text\"}]}"
                    .getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, body.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(body);
            }
        });
        mockNode.createContext("/api/show", exchange -> {
            byte[] body = "{\"capabilities\":[\"completion\"]}".getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, body.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(body);
            }
        });
        mockNode.start();
        Map<String, Object> originalSettings = res.jsonPath().getMap("settings");
        String keyId = null;
        try {
            Response created = api.addProviderKey(Map.of(
                    "provider", "ollama",
                    "label", "Mock Live Ollama",
                    "base_url", "http://127.0.0.1:" + mockNode.getAddress().getPort(),
                    "verify", true
            ));
            assertEquals(200, created.statusCode(), created.asString());
            keyId = created.jsonPath().getString("key.id");
            assertEquals(2, created.jsonPath().getInt("key.discovered_count"));

            List<Map<String, Object>> liveCatalog = created.jsonPath().getList("catalog");
            List<String> ollamaIds = liveCatalog.stream()
                    .filter(m -> "ollama".equals(m.get("provider")))
                    .map(m -> String.valueOf(m.get("id")))
                    .toList();
            assertTrue(ollamaIds.contains("qwen2.5:14b"));
            assertFalse(ollamaIds.contains("qwen2.5:7b"), "Unverified static seed must not remain after live discovery");
            assertFalse(ollamaIds.contains("qwen2.5-coder:7b"), "Code-only model must be filtered out");
            assertFalse(ollamaIds.contains("nomic-embed-text"), "Embedding model must be filtered out");

            final String expectedKeyId = keyId;
            Map<String, Object> q14 = liveCatalog.stream()
                    .filter(m -> "ollama".equals(m.get("provider")) && "qwen2.5:14b".equals(m.get("id")))
                    .findFirst()
                    .orElseThrow();
            assertEquals(List.of("translation", "glossary"), q14.get("tasks"));
            assertEquals(List.of(expectedKeyId), q14.get("key_ids"));

            Map<String, Object> qSmall = liveCatalog.stream()
                    .filter(m -> "ollama".equals(m.get("provider")) && "qwen2.5:1.5b".equals(m.get("id")))
                    .findFirst()
                    .orElseThrow();
            assertEquals(List.of(), qSmall.get("tasks"));
        } finally {
            mockNode.stop(0);
            if (keyId != null) {
                api.deleteProviderKey(keyId);
            }
            api.updateSettings(originalSettings);
        }
    }
}
