package com.subbuild.qa.api;

import io.restassured.RestAssured;
import io.restassured.builder.RequestSpecBuilder;
import io.restassured.http.ContentType;
import io.restassured.response.Response;
import io.restassured.specification.RequestSpecification;

import java.io.File;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * RestAssured API client for SubBuild Studio.
 * Automatically injects the X-SubBuild-Token session header and supports both
 * direct REST contract assertions and rapid test-data seeding for Hybrid API+UI tests.
 */
public class SubBuildApiClient {

    public static final String TOKEN_HEADER = "X-SubBuild-Token";
    private static final Set<String> TERMINAL_STATUSES = Set.of("COMPLETED", "FAILED", "CANCELLED");

    private final String baseUrl;
    private final String sessionToken;

    public SubBuildApiClient(String baseUrl, String sessionToken) {
        this.baseUrl = baseUrl;
        this.sessionToken = sessionToken;
    }

    private RequestSpecification authed() {
        return new RequestSpecBuilder()
                .setBaseUri(baseUrl)
                .addHeader(TOKEN_HEADER, sessionToken)
                .setContentType(ContentType.JSON)
                .build();
    }

    private RequestSpecification authedNoBodyType() {
        return new RequestSpecBuilder()
                .setBaseUri(baseUrl)
                .addHeader(TOKEN_HEADER, sessionToken)
                .build();
    }

    private RequestSpecification anonymous() {
        return new RequestSpecBuilder()
                .setBaseUri(baseUrl)
                .setContentType(ContentType.JSON)
                .build();
    }

    // -------------------------------------------------------------------------
    // System & Security
    // -------------------------------------------------------------------------

    public Response getSystemInfo() {
        return RestAssured.given().spec(authed()).get("/api/system/info");
    }

    public Response getWithoutToken(String path) {
        return RestAssured.given().spec(anonymous()).get(path);
    }

    public Response sendCrossOrigin(String method, String path, String origin, Object body) {
        RequestSpecification spec = RestAssured.given()
                .spec(authed())
                .header("Origin", origin);
        if (body != null) {
            spec = spec.body(body);
        }
        return switch (method.toUpperCase()) {
            case "POST" -> spec.post(path);
            case "PUT" -> spec.put(path);
            case "PATCH" -> spec.patch(path);
            case "DELETE" -> spec.delete(path);
            default -> throw new IllegalArgumentException("Unsupported method: " + method);
        };
    }

    // -------------------------------------------------------------------------
    // Media
    // -------------------------------------------------------------------------

    public Response probePath(String filePath) {
        return RestAssured.given().spec(authed())
                .body(Map.of("file_path", filePath))
                .post("/api/media/probe-path");
    }

    public Response parseSubtitles(File srtFile, String encoding) {
        RequestSpecification spec = RestAssured.given()
                .spec(authedNoBodyType())
                .multiPart("file", srtFile);
        if (encoding != null && !encoding.isBlank()) {
            spec = spec.queryParam("encoding", encoding);
        }
        return spec.post("/api/media/parse-subtitles");
    }

    // -------------------------------------------------------------------------
    // Jobs & Segments
    // -------------------------------------------------------------------------

    public Response listJobs() {
        return RestAssured.given().spec(authed()).get("/api/jobs");
    }

    public Response getJob(String jobId) {
        return RestAssured.given().spec(authed()).get("/api/jobs/{jobId}", jobId);
    }

    public Response createJob(Map<String, Object> payload) {
        return RestAssured.given().spec(authed()).body(payload).post("/api/jobs");
    }

    public Response deleteJob(String jobId) {
        return RestAssured.given().spec(authed()).delete("/api/jobs/{jobId}", jobId);
    }

    public Response bulkDeleteJobs(List<String> jobIds) {
        return RestAssured.given().spec(authed())
                .body(Map.of("job_ids", jobIds))
                .post("/api/jobs/bulk-delete");
    }

    public Response deleteAllJobs() {
        return RestAssured.given().spec(authed()).delete("/api/jobs");
    }

    public Response updateJobSettings(String jobId, Map<String, Object> payload) {
        return RestAssured.given().spec(authed())
                .body(payload)
                .patch("/api/jobs/{jobId}/settings", jobId);
    }

    public Response createSegment(String jobId, Map<String, Object> payload) {
        return RestAssured.given().spec(authed())
                .body(payload)
                .post("/api/jobs/{jobId}/segments", jobId);
    }

    public Response patchSegment(String jobId, long segmentId, Map<String, Object> payload) {
        return RestAssured.given().spec(authed())
                .body(payload)
                .patch("/api/jobs/{jobId}/segments/{segmentId}", jobId, segmentId);
    }

    public Response deleteSegment(String jobId, long segmentId) {
        return RestAssured.given().spec(authed())
                .delete("/api/jobs/{jobId}/segments/{segmentId}", jobId, segmentId);
    }

    public Response getHistory(String jobId) {
        return RestAssured.given().spec(authed()).get("/api/jobs/{jobId}/history", jobId);
    }

    public Response undoHistory(String jobId) {
        return RestAssured.given().spec(authed()).post("/api/jobs/{jobId}/history/undo", jobId);
    }

    public Response redoHistory(String jobId) {
        return RestAssured.given().spec(authed()).post("/api/jobs/{jobId}/history/redo", jobId);
    }

    public Response runQaCheck(String jobId) {
        return RestAssured.given().spec(authed()).get("/api/jobs/{jobId}/qa", jobId);
    }

    public Response splitSegment(String jobId, long segmentId, Double splitTime) {
        Map<String, Object> body = splitTime != null ? Map.of("split_time", splitTime) : Map.of();
        return RestAssured.given().spec(authed())
                .body(body)
                .post("/api/jobs/{jobId}/segments/{segmentId}/split", jobId, segmentId);
    }

    public Response mergeSegments(String jobId, List<Long> segmentIds) {
        return RestAssured.given().spec(authed())
                .body(Map.of("segment_ids", segmentIds))
                .post("/api/jobs/{jobId}/segments/merge", jobId);
    }

    public Response shiftSegments(String jobId, double offsetSeconds, List<Long> segmentIds) {
        Map<String, Object> body = segmentIds != null
                ? Map.of("offset_seconds", offsetSeconds, "segment_ids", segmentIds)
                : Map.of("offset_seconds", offsetSeconds);
        return RestAssured.given().spec(authed())
                .body(body)
                .post("/api/jobs/{jobId}/segments/shift", jobId);
    }

    public Response updateGlossary(String jobId, Map<String, Object> payload) {
        return RestAssured.given().spec(authed())
                .body(payload)
                .put("/api/jobs/{jobId}/glossary", jobId);
    }

    public Response listGlossaryProfiles() {
        return RestAssured.given().spec(authed()).get("/api/glossary-profiles");
    }

    public Response retranslateJob(String jobId, Map<String, Object> payload) {
        return RestAssured.given().spec(authed())
                .body(payload)
                .post("/api/jobs/{jobId}/translate", jobId);
    }

    public Response exportJobLog(String jobId) {
        return RestAssured.given().spec(authed()).get("/api/jobs/{jobId}/export/log", jobId);
    }

    public Response startEmbedVideo(String jobId, Map<String, Object> payload) {
        return RestAssured.given().spec(authed())
                .body(payload)
                .post("/api/jobs/{jobId}/export/embed", jobId);
    }

    public Response getEmbedStatus(String jobId, String taskId) {
        return RestAssured.given().spec(authed())
                .get("/api/jobs/{jobId}/export/embed/{taskId}", jobId, taskId);
    }

    // -------------------------------------------------------------------------
    // Export
    // -------------------------------------------------------------------------

    public Response exportSrt(String jobId, boolean useTranslated, boolean applyRtl, boolean bilingual) {
        return RestAssured.given().spec(authed())
                .queryParam("use_translated", useTranslated)
                .queryParam("apply_rtl", applyRtl)
                .queryParam("bilingual", bilingual)
                .get("/api/jobs/{jobId}/export/srt", jobId);
    }

    public Response exportVtt(String jobId, boolean useTranslated, boolean applyRtl) {
        return RestAssured.given().spec(authed())
                .queryParam("use_translated", useTranslated)
                .queryParam("apply_rtl", applyRtl)
                .get("/api/jobs/{jobId}/export/vtt", jobId);
    }

    // -------------------------------------------------------------------------
    // Settings
    // -------------------------------------------------------------------------

    public Response getSettings() {
        return RestAssured.given().spec(authed()).get("/api/settings");
    }

    public Response updateSettings(Map<String, Object> payload) {
        return RestAssured.given().spec(authed()).body(payload).put("/api/settings");
    }

    public Response estimateCost(Map<String, Object> payload) {
        return RestAssured.given().spec(authed()).body(payload).post("/api/settings/estimate");
    }

    public Response listProviderKeys() {
        return RestAssured.given().spec(authed()).get("/api/settings/keys");
    }

    public Response addProviderKey(Map<String, Object> payload) {
        return RestAssured.given().spec(authed()).body(payload).post("/api/settings/keys");
    }

    public Response updateProviderKey(String keyId, Map<String, Object> payload) {
        return RestAssured.given().spec(authed()).body(payload).put("/api/settings/keys/{keyId}", keyId);
    }

    public Response deleteProviderKey(String keyId) {
        return RestAssured.given().spec(authed()).delete("/api/settings/keys/{keyId}", keyId);
    }

    // -------------------------------------------------------------------------
    // Hybrid API + UI Seeding Helpers
    // -------------------------------------------------------------------------

    /**
     * Polls GET /api/jobs/{jobId} until it reaches a terminal state.
     * Throws AssertionError if the timeout expires.
     */
    public Response waitForJobCompletion(String jobId, Duration timeout) {
        Instant deadline = Instant.now().plus(timeout);
        while (Instant.now().isBefore(deadline)) {
            Response response = getJob(jobId);
            if (response.statusCode() == 200) {
                String status = response.jsonPath().getString("status");
                if (TERMINAL_STATUSES.contains(status)) {
                    return response;
                }
            }
            LockSupportWait.pauseMillis(100);
        }
        throw new AssertionError("Timed out waiting for job " + jobId + " to reach a terminal status");
    }

    /**
     * Seeds a completed job with real subtitle segments using an MKV carrying an embedded SRT track.
     */
    public String seedCompletedEmbeddedJob(String videoPath) {
        Response created = createJob(Map.of(
                "video_path", videoPath,
                "use_embedded_subtitles", true,
                "supplement_with_whisper", false,
                "detect_onscreen_text", false,
                "enable_translation", false
        ));
        if (created.statusCode() != 200) {
            throw new IllegalStateException("Failed to seed job: " + created.statusCode() + " " + created.asString());
        }
        String jobId = created.jsonPath().getString("id");
        Response finished = waitForJobCompletion(jobId, Duration.ofSeconds(20));
        String status = finished.jsonPath().getString("status");
        if (!"COMPLETED".equals(status)) {
            throw new IllegalStateException("Seeded job did not complete: " + status + " - " + finished.asString());
        }
        return jobId;
    }

    private static final class LockSupportWait {
        private static void pauseMillis(long millis) {
            java.util.concurrent.locks.LockSupport.parkNanos(millis * 1_000_000L);
        }
    }
}
