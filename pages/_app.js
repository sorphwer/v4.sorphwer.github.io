import '@/css/tailwind.css'
import '@/css/prism.css'
import '@/css/mermaid.css'
import '@/css/notion.css'
import 'katex/dist/katex.css'
import '@fortawesome/fontawesome-svg-core/styles.css'

import '@fontsource/inter/variable-full.css'
import 'react-notion-x/src/styles.css'
import 'prismjs/themes/prism-tomorrow.css'
import 'katex/dist/katex.min.css'

import { ThemeProvider } from 'next-themes'
import Head from 'next/head'

import siteMetadata from '@/data/siteMetadata'
import Analytics from '@/components/analytics'
import LayoutWrapper from '@/components/LayoutWrapper'
import { ClientReload } from '@/components/ClientReload'
import { UserProvider } from '@auth0/nextjs-auth0/client'

import { library } from '@fortawesome/fontawesome-svg-core'
import {
  faTags,
  faEdit,
  faSun,
  faSnowflake,
  faFan,
  faLeaf,
} from '@fortawesome/free-solid-svg-icons'

import { config } from '@fortawesome/fontawesome-svg-core'
config.autoAddCss = false
library.add(faTags, faEdit, faSun, faSnowflake, faFan, faLeaf)

const isDevelopment = process.env.NODE_ENV === 'development'
const isSocket = process.env.SOCKET

export default function App({ Component, pageProps }) {
  return (
    <ThemeProvider attribute="class" defaultTheme={siteMetadata.theme}>
      <Head>
        <meta content="width=device-width, initial-scale=1" name="viewport" />
      </Head>
      {isDevelopment && isSocket && <ClientReload />}
      <Analytics />
      <UserProvider>
        <LayoutWrapper>
          <Component {...pageProps} />
        </LayoutWrapper>
      </UserProvider>
    </ThemeProvider>
  )
}
