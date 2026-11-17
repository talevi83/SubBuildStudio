package com.subbuild.qa.tests.ui;

import com.subbuild.qa.base.BaseTest;
import com.subbuild.qa.pages.ExportToolbarComponent;
import com.subbuild.qa.pages.IngestPage;
import com.subbuild.qa.pages.StudioEditorPage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Studio Editor E2E Tests (Hybrid API Seeding + Selenium UI)")
class StudioEditorUiTest extends BaseTest {

    private String seededJobId;

    @BeforeEach
    void seedCompletedJobViaApi() {
        seededJobId = api.seedCompletedEmbeddedJob(sampleVideoPath.toString());
    }

    @AfterEach
    void cleanupJobs() {
        api.deleteAllJobs();
    }

    private StudioEditorPage openSeededJobInStudio() {
        IngestPage ingest = openStudio();
        ingest.sidebar().waitForJobCard(seededJobId);
        ingest.sidebar().selectJobById(seededJobId);
        return new StudioEditorPage(driver)
                .waitForWorkspaceLoaded()
                .waitForCompletedStatus();
    }

    @Test
    @Tag("smoke")
    @Tag("regression")
    @Tag("ui")
    @DisplayName("Open API-seeded job in Studio, inspect toolbar controls, and persist live onscreen position")
    void openSeededJobAndUpdateOnscreenPosition() {
        StudioEditorPage studio = openSeededJobInStudio();
        assertEquals(3, studio.getVisibleSubtitleRowCount());

        ExportToolbarComponent toolbar = studio.exportToolbar();
        assertEquals("top_center", toolbar.getOnscreenPosition());
        assertTrue(toolbar.isOriginalSrtEnabled());
        assertTrue(toolbar.isWebVttEnabled());
        assertTrue(toolbar.isJobLogVisible());
        assertTrue(toolbar.isGlossaryEnabled());
        assertTrue(toolbar.isEmbedEnabled());

        toolbar.setOnscreenPosition("middle_right");

        // Verify live PATCH persisted in backend DB via RestAssured
        String persisted = api.getJob(seededJobId).jsonPath().getString("onscreen_position");
        assertEquals("middle_right", persisted);
    }

    @Test
    @Tag("regression")
    @Tag("ui")
    @DisplayName("Insert subtitle segment, Search & Replace, Undo/Redo history, QA panel, and Sidebar Clear All")
    void insertReplaceUndoRedoQaAndSidebarCleanup() {
        StudioEditorPage studio = openSeededJobInStudio();
        assertEquals(3, studio.getVisibleSubtitleRowCount());

        // Insert a 4th line via modal
        studio.insertSegment("00:00:03,850", "00:00:03,980", "Fourth QA line.");
        assertEquals(4, studio.getVisibleSubtitleRowCount());
        assertTrue(studio.isUndoEnabled());

        // Search & Replace across original column
        studio.replaceAllInOriginal("Fourth QA line.", "Replaced QA line.");
        assertEquals("Replaced QA line.", studio.getOriginalTextAtRow(3));

        // Undo reverts Replace All; Redo restores Replace All
        studio.undo();
        assertTrue(studio.isRedoEnabled());
        studio.redo();
        assertEquals("Replaced QA line.", studio.getOriginalTextAtRow(3));

        // History panel lists recorded steps
        int historyCount = studio.openHistoryAndCountEntries();
        assertTrue(historyCount >= 3, "Expected original + insert + bulk_edit history entries");

        // Model-free QA check renders panel cleanly
        String qaText = studio.runQualityCheck();
        assertFalse(qaText.isBlank());

        // Clean up job from Sidebar UI
        studio.sidebar().deleteAllViaClearAll();
        assertEquals(0, studio.sidebar().jobCount());
    }
}
