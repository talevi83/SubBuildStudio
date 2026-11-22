import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// Fonts are bundled with the app (no request to Google Fonts: works offline, nothing leaks the IP)
import '@fontsource/inter/300.css'
import '@fontsource/inter/400.css'
import '@fontsource/inter/500.css'
import '@fontsource/inter/600.css'
import '@fontsource/inter/700.css'
import '@fontsource/heebo/300.css'
import '@fontsource/heebo/400.css'
import '@fontsource/heebo/500.css'
import '@fontsource/heebo/600.css'
import '@fontsource/heebo/700.css'
import '@fontsource/jetbrains-mono/400.css'
import '@fontsource/jetbrains-mono/500.css'
import '@fontsource/jetbrains-mono/600.css'
import '@fontsource/rubik/400.css'
import '@fontsource/rubik/500.css'
import '@fontsource/rubik/700.css'
import '@fontsource/assistant/400.css'
import '@fontsource/assistant/600.css'
import '@fontsource/assistant/700.css'
import '@fontsource/secular-one/400.css'
import '@fontsource/alef/400.css'
import '@fontsource/alef/700.css'
import '@fontsource/varela-round/400.css'
import '@fontsource/frank-ruhl-libre/400.css'
import '@fontsource/frank-ruhl-libre/700.css'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
