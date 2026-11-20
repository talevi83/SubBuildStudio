; SubBuild Studio - Windows installer (Inno Setup 6)
;
; Build (after packaging\windows\stage.ps1 has filled build\windows\app):
;   ISCC.exe /DAppVersion=2.2.0 packaging\windows\SubBuildStudio.iss
; Output: build\windows\SubBuild-Studio-Setup.exe
;
; Installs per user (no administrator rights) to %LOCALAPPDATA%\Programs\SubBuild Studio.
; Projects, settings and API keys live in %LOCALAPPDATA%\SubBuild Studio (packaging\launcher.py),
; so reinstalling, updating or uninstalling never touches them.

#define AppName "SubBuild Studio"
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef StageDir
  #define StageDir "..\..\build\windows\app"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\build\windows"
#endif

[Setup]
; Never change AppId: Windows uses it to recognise an existing installation when updating
AppId={{8F3B2C1E-6A4D-4E7B-9C21-5D0B7A1F2E43}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=talevi83
AppPublisherURL=https://subbuildstudio.com
AppSupportURL=https://github.com/talevi83/SubBuildStudio/issues
AppUpdatesURL=https://github.com/talevi83/SubBuildStudio/releases
DefaultDirName={localappdata}\Programs\{#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir={#OutputDir}
OutputBaseFilename=SubBuild-Studio-Setup
SetupIconFile={#StageDir}\packaging\icon.ico
UninstallDisplayIcon={app}\packaging\icon.ico
UninstallDisplayName={#AppName}
LicenseFile={#StageDir}\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
CloseApplications=yes
ShowLanguageDialog=auto

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "he"; MessagesFile: "compiler:Languages\Hebrew.isl"

[CustomMessages]
en.SetupEnv=Installing the AI engine (Python packages). This happens once and can take several minutes...
he.SetupEnv=מתקין את מנוע הבינה המלאכותית (חבילות Python). זה קורה פעם אחת ויכול לקחת כמה דקות...
en.SetupEnvFailed=Installing the Python packages did not finish (code %1).%n%nCheck the internet connection and run the installer again.
he.SetupEnvFailed=התקנת חבילות ה-Python לא הסתיימה (קוד %1).%n%nבדקו את החיבור לאינטרנט והריצו את ההתקנה שוב.
en.LaunchApp=Start SubBuild Studio
he.LaunchApp=הפעלת SubBuild Studio

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; Remove the previous version's code first, so files deleted in a new version do not linger
Type: filesandordirs; Name: "{app}\backend\app"
Type: filesandordirs; Name: "{app}\frontend"

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\packaging\launcher.py"""; WorkingDir: "{app}"; IconFilename: "{app}\packaging\icon.ico"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\packaging\launcher.py"""; WorkingDir: "{app}"; IconFilename: "{app}\packaging\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\packaging\launcher.py"""; WorkingDir: "{app}"; Description: "{cm:LaunchApp}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Created after installation (the packages installed into python\, caches)
Type: filesandordirs; Name: "{app}\backend"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\.uv-cache"
Type: filesandordirs; Name: "{app}\frontend"
Type: filesandordirs; Name: "{app}\packaging"
Type: filesandordirs; Name: "{app}\tools"
Type: dirifempty; Name: "{app}"

[Code]
// After the files are copied: create / update the Python environment (packaging\windows\setup-env.cmd).
// It runs in a visible console window so the user sees the download progress.
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    WizardForm.StatusLabel.Caption := CustomMessage('SetupEnv');
    if (not Exec(ExpandConstant('{cmd}'),
                 '/c ""' + ExpandConstant('{app}\packaging\windows\setup-env.cmd') + '""',
                 ExpandConstant('{app}'), SW_SHOWNORMAL, ewWaitUntilTerminated, ResultCode))
       or (ResultCode <> 0) then
      MsgBox(FmtMessage(CustomMessage('SetupEnvFailed'), [IntToStr(ResultCode)]), mbError, MB_OK);
  end;
end;
