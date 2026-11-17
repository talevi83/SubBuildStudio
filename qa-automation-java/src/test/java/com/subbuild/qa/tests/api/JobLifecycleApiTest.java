package com.subbuild.qa.tests.api;

import com.subbuild.qa.base.BaseApiTest;
import io.restassured.response.Response;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Job Lifecycle, Segment Editing, History & Export API Tests")
class JobLifecycleApiTest extends BaseApiTest {

    @AfterEach
    void cleanupJobs() {
        api.deleteAllJobs();
    }

    @Test
    @Tag("smoke")
    @Tag("regression")
    @Tag("api")
    @DisplayName("Probe embedded MKV, run pipeline to completion, verify SRT/VTT exports, and delete")
    void embeddedJobLifecycleAndExport() {
        Response probe = api.probePath(sampleVideoPath.toString());
        assertEquals(200, probe.statusCode());
        assertFalse(probe.jsonPath().getList("subtitle_streams").isEmpty(), "Sample MKV must expose embedded SRT");

        String jobId = api.seedCompletedEmbeddedJob(sampleVideoPath.toString());
        Response job = api.getJob(jobId);
        assertEquals("COMPLETED", job.jsonPath().getString("status"));
        List<Map<String, Object>> segments = job.jsonPath().getList("segments");
        assertEquals(3, segments.size());
        assertEquals("Hello from SubBuild QA.", segments.get(0).get("original_text"));

        Response srt = api.exportSrt(jobId, false, false, false);
        assertEquals(200, srt.statusCode());
        assertTrue(srt.asString().contains("Hello from SubBuild QA."));
        assertTrue(srt.getHeader("Content-Disposition").contains(".srt"));

        Response vtt = api.exportVtt(jobId, false, false);
        assertEquals(200, vtt.statusCode());
        assertTrue(vtt.asString().startsWith("WEBVTT"));
        assertTrue(vtt.asString().contains("Everything is synced cleanly."));

        Response deleted = api.deleteJob(jobId);
        assertEquals(200, deleted.statusCode());
        assertEquals(404, api.getJob(jobId).statusCode());
    }

    @Test
    @Tag("regression")
    @Tag("api")
    @DisplayName("Segment CRUD, edit history undo/redo, live onscreen position, and QA check")
    void segmentCrudHistoryAndQaFlow() {
        String jobId = api.seedCompletedEmbeddedJob(sampleVideoPath.toString());

        // Insert a 4th segment after the 3rd cue
        Response created = api.createSegment(jobId, Map.of(
                "start_time", 3.85,
                "end_time", 3.98,
                "original_text", "Injected QA cue."
        ));
        assertEquals(200, created.statusCode());
        long createdId = created.jsonPath().getLong("created_id");
        assertEquals(4, created.jsonPath().getList("segments").size());

        // Patch text
        Response patched = api.patchSegment(jobId, createdId, Map.of("original_text", "Updated QA cue."));
        assertEquals(200, patched.statusCode());
        assertEquals("Updated QA cue.", patched.jsonPath().getString("original_text"));

        // Undo -> reverts text edit; Undo again -> reverts insert back to 3 segments
        Response undo1 = api.undoHistory(jobId);
        assertEquals(200, undo1.statusCode());
        Response undo2 = api.undoHistory(jobId);
        assertEquals(200, undo2.statusCode());
        assertEquals(3, undo2.jsonPath().getList("segments").size());

        // Redo -> restores the inserted segment
        Response redo = api.redoHistory(jobId);
        assertEquals(200, redo.statusCode());
        assertEquals(4, redo.jsonPath().getList("segments").size());

        // Update live onscreen position setting
        Response posUpdate = api.updateJobSettings(jobId, Map.of("onscreen_position", "middle_right"));
        assertEquals(200, posUpdate.statusCode());
        assertEquals("middle_right", posUpdate.jsonPath().getString("onscreen_position"));

        // Run model-free QA check
        Response qa = api.runQaCheck(jobId);
        assertEquals(200, qa.statusCode());

        // Parse subtitles endpoint sanity check
        Response parsed = api.parseSubtitles(sampleSrtPath.toFile(), null);
        assertEquals(200, parsed.statusCode());
        assertEquals(3, parsed.jsonPath().getInt("cue_count"));
    }
}
