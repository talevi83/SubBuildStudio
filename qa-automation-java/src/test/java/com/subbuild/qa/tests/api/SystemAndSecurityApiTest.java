package com.subbuild.qa.tests.api;

import com.subbuild.qa.base.BaseApiTest;
import io.restassured.response.Response;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("API & Security Contract Tests")
class SystemAndSecurityApiTest extends BaseApiTest {

    @Test
    @Tag("smoke")
    @Tag("regression")
    @Tag("api")
    @Tag("security")
    @DisplayName("GET /api/system/info returns model catalog and security headers when authenticated")
    void systemInfoReturnsCatalogAndSecurityHeaders() {
        Response res = api.getSystemInfo();

        assertEquals(200, res.statusCode());
        List<String> whisperModels = res.jsonPath().getList("whisper_models");
        List<String> geminiModels = res.jsonPath().getList("gemini_models");
        assertTrue(whisperModels.contains("large-v3-turbo"), "Expected large-v3-turbo in whisper_models");
        assertFalse(geminiModels.isEmpty(), "Expected non-empty gemini_models");

        assertEquals("DENY", res.getHeader("X-Frame-Options"));
        assertEquals("nosniff", res.getHeader("X-Content-Type-Options"));
        String csp = res.getHeader("Content-Security-Policy");
        assertNotNull(csp);
        assertTrue(csp.contains("frame-ancestors 'none'"), "CSP must deny framing");
    }

    @ParameterizedTest(name = "Anonymous GET {0} returns 401")
    @ValueSource(strings = {"/api/jobs", "/api/system/info", "/api/settings", "/openapi.json"})
    @Tag("smoke")
    @Tag("regression")
    @Tag("api")
    @Tag("security")
    @DisplayName("Protected API endpoints reject requests missing X-SubBuild-Token")
    void protectedEndpointsRequireSessionToken(String endpoint) {
        Response res = api.getWithoutToken(endpoint);
        assertEquals(401, res.statusCode());
    }

    @ParameterizedTest(name = "Cross-site {0} {1} from untrusted Origin is blocked with 403")
    @CsvSource({
            "POST, /api/jobs/bulk-delete",
            "PUT, /api/settings",
            "DELETE, /api/jobs"
    })
    @Tag("regression")
    @Tag("api")
    @Tag("security")
    @DisplayName("CSRF guard blocks state-changing requests from untrusted Origin")
    void crossOriginStateChangesAreBlocked(String method, String path) {
        Map<String, Object> body = "POST".equals(method) ? Map.of("delete_all", false) : Map.of();
        Response res = api.sendCrossOrigin(method, path, "https://evil.example", body);
        assertEquals(403, res.statusCode());
    }

    @ParameterizedTest(name = "Network UNC path rejected: {0}")
    @ValueSource(strings = {
            "\\\\attacker\\share\\movie.mp4",
            "//attacker/share/movie.mp4",
            "\\\\?\\UNC\\attacker\\share\\movie.mp4"
    })
    @Tag("regression")
    @Tag("api")
    @Tag("security")
    @DisplayName("Media probe rejects UNC/network paths to prevent NTLM leaks")
    void networkPathsAreRejected(String uncPath) {
        Response res = api.probePath(uncPath);
        assertEquals(400, res.statusCode());
        assertTrue(res.jsonPath().getString("detail").contains("Network paths"));
    }

    @ParameterizedTest(name = "Cost estimate for {0}s (cost_saver={1})")
    @CsvSource({
            "1800, false",
            "1800, true",
            "3600, false"
    })
    @Tag("regression")
    @Tag("api")
    @DisplayName("POST /api/settings/estimate returns positive breakdown and Cost-Saver reduces vision cost")
    void costEstimateCalculation(int durationSeconds, boolean costSaver) {
        Response res = api.estimateCost(Map.of(
                "duration_seconds", durationSeconds,
                "enable_translation", true,
                "detect_onscreen_text", true,
                "cost_saver_mode", costSaver
        ));
        assertEquals(200, res.statusCode());
        double total = res.jsonPath().getDouble("total_cost_usd");
        assertTrue(total > 0.0, "Total estimated cost should be positive");
        assertNotNull(res.jsonPath().getMap("stages.translation"));
        assertNotNull(res.jsonPath().getMap("stages.onscreen"));
    }
}
