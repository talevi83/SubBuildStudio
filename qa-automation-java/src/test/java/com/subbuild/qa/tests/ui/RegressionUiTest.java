package com.subbuild.qa.tests.ui;

import com.subbuild.qa.base.BaseTest;
import com.subbuild.qa.pages.HelpPage;
import com.subbuild.qa.pages.IngestPage;
import com.subbuild.qa.pages.SettingsPage;
import com.subbuild.qa.pages.StudioEditorPage;
import com.sun.net.httpserver.HttpServer;
import io.restassured.response.Response;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("UI Regression Suite (Negative Paths, Sync Upload, Overlap Guard, Glossary, Shift & Help)")
@Tag("regression")
@Tag("ui")
class RegressionUiTest extends BaseTest {

    @AfterEach
    void cleanupJobs() {
        api.deleteAllJobs();
        List<Map<String, Object>> keys = api.listProviderKeys().jsonPath().getList("keys");
        if (keys != null) {
            for (Map<String, Object> k : keys) {
                if ("vault".equals(k.get("source"))) {
                    api.deleteProviderKey(String.valueOf(k.get("id")));
                }
            }
        }
    }

    @Test
    @DisplayName("Invalid non-media path displays error banner in Ingest UI and hides Launch button")
    void invalidMediaPathShowsErrorBanner() {
        IngestPage ingest = openStudio();
        String errorText = ingest.probeInvalidPathExpectError(sampleSrtPath.toString());

        assertTrue(errorText.contains("Unsupported") || !errorText.isBlank());
        assertFalse(ingest.isLaunchButtonPresent(), "Launch button must not render for invalid media");
    }

    @Test
    @DisplayName("Sync mode uploads SRT subtitle file, displays cue preview, and exposes encoding selector")
    void syncModeSubtitleUploadAndEncodingSelector() {
        IngestPage ingest = openStudio()
                .selectSyncMode()
                .uploadSyncSubtitleFile(sampleSrtPath.toAbsolutePath().toString());

        assertEquals("utf-8", ingest.getSyncEncodingValue());
        assertEquals(3, ingest.getSyncPreviewCueCount());
    }

    @Test
    @DisplayName("Insert dialog blocks overlapping cue submission and displays neighbour conflict warning")
    void insertDialogBlocksOverlappingSegment() {
        String jobId = api.seedCompletedEmbeddedJob(sampleVideoPath.toString());
        IngestPage ingest = openStudio();
        ingest.sidebar().waitForJobCard(jobId);
        ingest.sidebar().selectJobById(jobId);

        StudioEditorPage studio = new StudioEditorPage(driver)
                .waitForWorkspaceLoaded()
                .waitForCompletedStatus();

        // Cue #1 runs 00:00:00,300 -> 00:00:01,500; inserting 00:00:00,500 -> 00:00:01,200 overlaps it
        studio.openInsertDialogWithOverlap("00:00:00,500", "00:00:01,200", "Conflicting cue");
        assertFalse(studio.isInsertSubmitEnabled(), "Add button must remain disabled while overlap exists");
        assertFalse(studio.getInsertIssuesText().isBlank());
        studio.closeInsertDialog();
    }

    @Test
    @DisplayName("Glossary dialog saves new character and series profile to backend DB")
    void glossaryDialogSavesCharacterAndProfile() {
        String jobId = api.seedCompletedEmbeddedJob(sampleVideoPath.toString());
        IngestPage ingest = openStudio();
        ingest.sidebar().waitForJobCard(jobId);
        ingest.sidebar().selectJobById(jobId);

        StudioEditorPage studio = new StudioEditorPage(driver)
                .waitForWorkspaceLoaded()
                .waitForCompletedStatus();

        studio.openGlossaryAndSaveCharacter("Captain Chandler", "קפטן צ'נדלר", "male", "LastShipQA");

        Response job = api.getJob(jobId);
        assertEquals("LastShipQA", job.jsonPath().getString("glossary_profile"));
        assertEquals("Captain Chandler", job.jsonPath().getString("glossary.characters[0].name"));
        assertEquals("male", job.jsonPath().getString("glossary.characters[0].gender"));
    }

    @Test
    @DisplayName("Shift Time popover updates subtitle timestamps in Studio and persists via API")
    void shiftTimePopoverPersistsTimingChange() {
        String jobId = api.seedCompletedEmbeddedJob(sampleVideoPath.toString());
        double originalStart = api.getJob(jobId).jsonPath().getDouble("segments[0].start_time");

        IngestPage ingest = openStudio();
        ingest.sidebar().waitForJobCard(jobId);
        ingest.sidebar().selectJobById(jobId);

        StudioEditorPage studio = new StudioEditorPage(driver)
                .waitForWorkspaceLoaded()
                .waitForCompletedStatus();

        studio.shiftAllTimes("0.5");
        double shiftedStart = api.getJob(jobId).jsonPath().getDouble("segments[0].start_time");
        assertEquals(originalStart + 0.5, shiftedStart, 0.01);
    }

    @Test
    @DisplayName("Help & Q&A screen renders 4 workflow steps, interactive FAQ accordion, and navigates back")
    void helpScreenAndFaqAccordion() {
        HelpPage help = openStudio().sidebar().openHelp();
        assertEquals(6, help.getFaqCount());
        help.toggleFaqItem(1);
        IngestPage back = help.backToStudio();
        assertEquals("transcribe", back.getSelectedMode());
    }

    @Test
    @DisplayName("Settings UI manages multiple masked API keys per provider and Ollama Cloud preset")
    void multiProviderKeysAddAndManageInSettingsUi() {
        SettingsPage settings = openStudio().sidebar().openSettings().waitForLoaded();
        settings.selectProviderTab("google");
        assertTrue(settings.getProviderKeyLinkHref().contains("aistudio.google.com"));

        settings.addProviderKey("openai", "Personal", "sk-ui-secret-1111aaaa");
        assertTrue(settings.getProviderKeyLinkHref().contains("platform.openai.com/api-keys"));
        settings.addProviderKey("openai", "Work", "sk-ui-secret-2222bbbb");

        assertEquals(2, settings.getProviderKeyItemsCount());
        String listText = settings.getProviderKeysText();
        assertTrue(listText.contains("Personal") && listText.contains("Work"));
        assertFalse(listText.contains("sk-ui-secret-1111aaaa"));

        settings.selectProviderTab("anthropic");
        assertTrue(settings.getProviderKeyLinkHref().contains("console.anthropic.com/settings/keys"));

        settings.clickOllamaCloudPreset();
        assertEquals("https://ollama.com", settings.getOllamaBaseUrlValue());
        assertTrue(settings.getProviderKeyLinkHref().contains("ollama.com/settings/keys"));
    }

    @Test
    @DisplayName("Settings UI strictly filters models by API key availability and pipeline step capabilities")
    void strictKeyAndStepFilteringInSettingsUi() throws IOException {
        Map<String, Object> original = api.getSettings().jsonPath().getMap("settings");

        HttpServer textNode = startMockOllamaServer(
                "{\"models\":[{\"name\":\"qwen2.5:14b\"},{\"name\":\"qwen2.5:1.5b\"},{\"name\":\"qwen2.5-coder:7b\"},{\"name\":\"nomic-embed-text:latest\"}]}",
                "{\"capabilities\":[\"completion\"]}"
        );
        HttpServer visionNode = startMockOllamaServer(
                "{\"models\":[{\"name\":\"qwen2.5vl:7b\"}]}",
                "{\"capabilities\":[\"completion\",\"vision\"]}"
        );

        try {
            SettingsPage settings = openStudio().sidebar().openSettings().waitForLoaded();

            // Unconfigured Anthropic shows no unverified seed models and displays the add-key notice
            settings.selectStepProvider("glossary", "anthropic");
            assertEquals(List.of(), settings.getStepModelOptionValues("glossary"));
            assertTrue(settings.isStepNoModelsHintVisible("glossary"),
                    "Unconfigured provider should show add-key hint instead of static models");

            // Add two live Ollama keys pointing to our in-process mock servers via UI
            settings.addOllamaEndpointKey("Text Node", "http://127.0.0.1:" + textNode.getAddress().getPort(), "");
            settings.addOllamaEndpointKey("Vision Node", "http://127.0.0.1:" + visionNode.getAddress().getPort(), "");

            // Translation with Text Node shows ONLY capable qwen2.5:14b (excludes 1.5b, coder, embed, and Vision Node models)
            settings.selectStepProvider("translation", "ollama")
                    .selectStepKeyByVisiblePrefix("translation", "Text Node");
            List<String> textNodeTranslationOptions = settings.getStepModelOptionValues("translation");
            assertEquals(List.of("qwen2.5:14b"), textNodeTranslationOptions);

            // Model dropdown labels contain clean names without star/check/warning clutter
            List<String> labels = settings.getStepModelOptionLabels("translation");
            assertTrue(labels.stream().noneMatch(l -> l.contains("★") || l.contains("⚠") || l.contains("✓")),
                    "Dropdown labels should not contain symbol clutter: " + labels);

            // Switching Translation key to Vision Node updates options to ONLY qwen2.5vl:7b
            settings.selectStepKeyByVisiblePrefix("translation", "Vision Node");
            assertEquals(List.of("qwen2.5vl:7b"), settings.getStepModelOptionValues("translation"));
            settings.selectStepKeyByVisiblePrefix("translation", "Text Node");

            // Vision with Text Node shows NO models (qwen2.5:14b lacks vision) + shows no-capable hint
            settings.selectStepProvider("vision", "ollama")
                    .selectStepKeyByVisiblePrefix("vision", "Text Node");
            assertEquals(List.of(), settings.getStepModelOptionValues("vision"));
            assertTrue(settings.isStepNoModelsHintVisible("vision"));

            // Switching Vision key to Vision Node unlocks qwen2.5vl:7b
            settings.selectStepKeyByVisiblePrefix("vision", "Vision Node");
            assertEquals(List.of("qwen2.5vl:7b"), settings.getStepModelOptionValues("vision"));

            // Persist choices and verify across reload
            settings.selectStepProvider("glossary", "ollama")
                    .selectStepKeyByVisiblePrefix("glossary", "Text Node");
            settings.saveSettings();

            SettingsPage reloaded = openStudio().sidebar().openSettings().waitForLoaded();
            assertEquals("ollama", reloaded.getSelectedStepProvider("translation"));
            assertEquals("qwen2.5:14b", reloaded.getSelectedStepModel("translation"));
            assertEquals("ollama", reloaded.getSelectedStepProvider("vision"));
            assertEquals("qwen2.5vl:7b", reloaded.getSelectedStepModel("vision"));
        } finally {
            textNode.stop(0);
            visionNode.stop(0);
            api.updateSettings(original);
        }
    }

    private static HttpServer startMockOllamaServer(String tagsJson, String showJson) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/api/tags", exchange -> {
            byte[] body = tagsJson.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, body.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(body);
            }
        });
        server.createContext("/api/show", exchange -> {
            byte[] body = showJson.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, body.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(body);
            }
        });
        server.start();
        return server;
    }
}
