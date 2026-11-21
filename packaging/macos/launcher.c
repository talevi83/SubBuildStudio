/*
 * SubBuild Studio - the macOS app's executable (Contents/MacOS/SubBuild Studio).
 *
 * Runs packaging/launcher.py on the Python that ships inside the app bundle, in this same process
 * (Python is embedded, not started as a child). Because the running executable lives in
 * Contents/MacOS, macOS treats the process as "SubBuild Studio": its name and icon appear in the
 * Dock and the menu bar instead of "python3.12".
 *
 * Bundle layout (built by packaging/macos/build-app.sh):
 *   Contents/MacOS/SubBuild Studio        this program
 *   Contents/Resources/python/            Python 3.12 + the app's packages
 *   Contents/Resources/app/               the application (backend/, frontend/dist, packaging/)
 *
 * Built with:  clang launcher.c -I<python>/include/python3.12 -L<python>/lib -lpython3.12
 */
#include <Python.h>

#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *what) {
    fprintf(stderr, "SubBuild Studio: %s\n", what);
}

static void join(char *out, size_t size, const char *a, const char *b) {
    snprintf(out, size, "%s/%s", a, b);
}

int main(int argc, char **argv) {
    char exe_raw[PATH_MAX];
    uint32_t exe_size = sizeof(exe_raw);
    if (_NSGetExecutablePath(exe_raw, &exe_size) != 0) {
        fail("executable path too long");
        return 1;
    }
    char exe[PATH_MAX];
    if (realpath(exe_raw, exe) == NULL) {
        fail("cannot resolve the executable path");
        return 1;
    }

    /* .../SubBuild Studio.app/Contents/MacOS/<exe>  ->  Contents, then the bundle itself */
    char macos_dir[PATH_MAX], contents[PATH_MAX], bundle[PATH_MAX];
    strlcpy(macos_dir, exe, sizeof(macos_dir));
    strlcpy(macos_dir, dirname(macos_dir), sizeof(macos_dir));
    strlcpy(contents, macos_dir, sizeof(contents));
    strlcpy(contents, dirname(contents), sizeof(contents));
    strlcpy(bundle, contents, sizeof(bundle));
    strlcpy(bundle, dirname(bundle), sizeof(bundle));

    char resources[PATH_MAX], python_home[PATH_MAX], python_exe[PATH_MAX], launcher[PATH_MAX];
    join(resources, sizeof(resources), contents, "Resources");
    join(python_home, sizeof(python_home), resources, "python");
    join(python_exe, sizeof(python_exe), python_home, "bin/python3.12");
    join(launcher, sizeof(launcher), resources, "app/packaging/launcher.py");

    /* A clean, predictable interpreter: ignore a Python set up in the user's shell, never write
       .pyc files into the (signed, often read-only) bundle, and use UTF-8 regardless of locale. */
    unsetenv("PYTHONHOME");
    unsetenv("PYTHONPATH");
    unsetenv("PYTHONSTARTUP");
    unsetenv("PYTHONEXECUTABLE");
    unsetenv("__PYVENV_LAUNCHER__");
    setenv("PYTHONNOUSERSITE", "1", 1);
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);
    setenv("PYTHONUTF8", "1", 1);
    /* Read by launcher.py: restart re-opens the app bundle rather than starting bare Python */
    setenv("SUBBUILD_APP_BUNDLE", bundle, 1);

    /* argv for Python: <python> <launcher.py> [user args...] (minus Finder's old -psn_ argument) */
    char **py_argv = calloc((size_t)argc + 2, sizeof(char *));
    if (py_argv == NULL) {
        fail("out of memory");
        return 1;
    }
    int py_argc = 0;
    py_argv[py_argc++] = python_exe;
    py_argv[py_argc++] = launcher;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "-psn_", 5) == 0) {
            continue;
        }
        py_argv[py_argc++] = argv[i];
    }

    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.write_bytecode = 0;
    config.user_site_directory = 0;

    status = PyConfig_SetBytesString(&config, &config.home, python_home);
    if (PyStatus_Exception(status)) goto error;
    /* sys.executable: the real interpreter, so anything that starts Python again still works */
    status = PyConfig_SetBytesString(&config, &config.executable, python_exe);
    if (PyStatus_Exception(status)) goto error;
    status = PyConfig_SetBytesString(&config, &config.program_name, python_exe);
    if (PyStatus_Exception(status)) goto error;
    status = PyConfig_SetBytesArgv(&config, py_argc, py_argv);
    if (PyStatus_Exception(status)) goto error;

    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) goto error;
    PyConfig_Clear(&config);
    return Py_RunMain();

error:
    PyConfig_Clear(&config);
    if (PyStatus_IsExit(status)) {
        return status.exitcode;
    }
    Py_ExitStatusException(status);
    return 1;
}
