package com.subbuild.qa.base;

import com.subbuild.qa.api.SubBuildApiClient;
import com.subbuild.qa.extensions.ExtentReportExtension;
import com.subbuild.qa.extensions.SubBuildServerExtension;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.extension.ExtendWith;

import java.nio.file.Path;

/**
 * Base class for API contract and hybrid seeding tests.
 * Provides an isolated backend server and an authenticated RestAssured SubBuildApiClient.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@ExtendWith({SubBuildServerExtension.class, ExtentReportExtension.class})
public abstract class BaseApiTest {

    protected String baseUrl;
    protected String sessionToken;
    protected Path sampleVideoPath;
    protected Path sampleSrtPath;
    protected SubBuildApiClient api;

    @BeforeAll
    void initApiContext(SubBuildServerExtension.ServerHandle handle) {
        this.baseUrl = handle.baseUrl();
        this.sessionToken = handle.token();
        this.sampleVideoPath = handle.sampleVideoPath();
        this.sampleSrtPath = handle.sampleSrtPath();
        this.api = new SubBuildApiClient(baseUrl, sessionToken);
    }
}
