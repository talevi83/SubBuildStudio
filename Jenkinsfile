pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
        timeout(time: 30, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '15', artifactNumToKeepStr: '10'))
    }

    parameters {
        choice(
            name: 'JAVA_PROFILE',
            choices: ['regression', 'smoke'],
            description: 'Java QA test profile to run (regression = all 52 tests, smoke = 9 critical-path tests)'
        )
        booleanParam(
            name: 'RUN_BACKEND_PYTEST',
            defaultValue: true,
            description: 'Run Python backend pytest suite (495 tests)'
        )
        booleanParam(
            name: 'RUN_FRONTEND_CHECKS',
            defaultValue: true,
            description: 'Run frontend lint (oxlint), typecheck (tsc), Vitest (11 tests), and Vite build'
        )
    }

    environment {
        CI = 'true'
        SUBBUILD_QA_HEADLESS = 'true'
        SUBBUILD_QA_TIMEOUT = '30'
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
    }

    stages {
        stage('Preflight Check') {
            steps {
                sh '''#!/usr/bin/env bash
                set -euo pipefail
                echo "Host: $(uname -a)"
                for cmd in python3 node npm java mvn ffmpeg; do
                    if ! command -v "$cmd" >/dev/null 2>&1; then
                        echo "ERROR: Required tool '$cmd' not found in PATH." >&2
                        exit 1
                    fi
                done

                echo "Python: $(python3 --version)"
                echo "Node:   $(node -v) | npm: $(npm -v)"
                echo "Java:   $(java -version 2>&1 | head -n 1)"
                echo "Maven:  $(mvn -version | head -n 1)"
                echo "FFmpeg: $(ffmpeg -version | head -n 1)"

                if command -v chromium >/dev/null 2>&1; then
                    echo "Browser: $(chromium --version)"
                elif command -v chromium-browser >/dev/null 2>&1; then
                    echo "Browser: $(chromium-browser --version)"
                elif command -v google-chrome >/dev/null 2>&1; then
                    echo "Browser: $(google-chrome --version)"
                else
                    echo "WARNING: No system Chromium/Chrome binary detected in PATH (Selenium Manager will be used on x86_64)."
                fi

                if command -v chromedriver >/dev/null 2>&1; then
                    echo "Driver:  $(chromedriver --version)"
                fi
                '''
            }
        }

        stage('Backend Setup & Pytest') {
            when {
                expression { return params.RUN_BACKEND_PYTEST }
            }
            steps {
                dir('backend') {
                    sh '''#!/usr/bin/env bash
                    set -euo pipefail
                    if [ ! -x ".venv/bin/python" ]; then
                        python3 -m venv .venv
                    fi

                    REQ_HASH=$(sha256sum requirements.txt | awk '{print $1}')
                    STAMP_FILE=".venv/.req.sha256"
                    if [ ! -f "$STAMP_FILE" ] || [ "$(cat "$STAMP_FILE")" != "$REQ_HASH" ]; then
                        grep -v '^nvidia-' requirements.txt > .venv/requirements-ci.txt
                        .venv/bin/python -m pip install --upgrade pip
                        .venv/bin/python -m pip install -r .venv/requirements-ci.txt
                        echo "$REQ_HASH" > "$STAMP_FILE"
                    fi

                    mkdir -p .pytest_cache
                    .venv/bin/python -m pytest -q --junitxml=.pytest_cache/pytest-junit.xml
                    '''
                }
            }
        }

        stage('Frontend Lint, Vitest & Build') {
            steps {
                dir('frontend') {
                    sh '''#!/usr/bin/env bash
                    set -euo pipefail
                    npm ci --no-audit --no-fund
                    if [ "${RUN_FRONTEND_CHECKS}" = "true" ]; then
                        npm run lint
                        npm run build
                    else
                        npx vite build
                    fi
                    '''
                }
            }
        }

        stage('Java Black-Box QA (RestAssured & Selenium)') {
            steps {
                dir('qa-automation-java') {
                    sh '''#!/usr/bin/env bash
                    set -euo pipefail
                    # Ensure backend/.venv exists even when RUN_BACKEND_PYTEST=false
                    if [ ! -x "../backend/.venv/bin/python" ]; then
                        (
                            cd ../backend
                            python3 -m venv .venv
                            grep -v '^nvidia-' requirements.txt > .venv/requirements-ci.txt
                            .venv/bin/python -m pip install -r .venv/requirements-ci.txt
                        )
                    fi

                    mvn -B clean test -P"${JAVA_PROFILE}" -Dsubbuild.headless=true -Dsubbuild.timeoutSeconds=30
                    '''
                }
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true,
                  testResults: 'backend/.pytest_cache/pytest-junit.xml,qa-automation-java/target/surefire-reports/*.xml'
            archiveArtifacts artifacts: 'qa-automation-java/target/reports/**,qa-automation-java/target/surefire-reports/**',
                             allowEmptyArchive: true,
                             fingerprint: true
            /* Optional: uncomment if the Jenkins HTML Publisher plugin is installed
            publishHTML([
                allowMissing: true,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'qa-automation-java/target/reports',
                reportFiles: 'subbuild-qa-report.html',
                reportName: 'SubBuild ExtentReports QA Dashboard'
            ])
            */
        }
    }
}
