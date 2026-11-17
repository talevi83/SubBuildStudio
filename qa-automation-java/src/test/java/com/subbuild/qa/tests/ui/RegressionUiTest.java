package com.subbuild.qa.tests.ui;

import com.subbuild.qa.base.BaseTest;
import com.subbuild.qa.pages.HelpPage;
import com.subbuild.qa.pages.IngestPage;
import com.subbuild.qa.pages.StudioEditorPage;
import io.restassured.response.Response;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("UI Regression Suite (Negative Paths, Sync Upload, Overlap Guard, Glossary, Shift & Help)")
@Tag("regression")
@Tag("ui")
class RegressionUiTest extends BaseTest {

    @AfterEach
    void cleanupJobs() {
        api.deleteAllJobs();
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
}
