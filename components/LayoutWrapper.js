import siteMetadata from '@/data/siteMetadata'
import headerNavLinks from '@/data/headerNavLinks'
import Logo from '@/data/logo.svg'
import Image from './Image'
import Link from './Link'
import SectionContainer from './SectionContainer'
import Footer from './Footer'
import MobileNav from './MobileNav'
import ThemeSwitch from './ThemeSwitch'
import { useRouter } from 'next/router'

const LayoutWrapper = ({ children }) => {
  const activeNavLinkClassNames = 'active'
  const nonActiveNavLinkClassNames = 'nonActive'
  const currentRoute = useRouter().pathname
  return (
    <SectionContainer>
      <header className="mt-10 flex items-center justify-between py-10">
        {/* <div>
            <Link href="/" aria-label={siteMetadata.headerTitle}>
              <div className="flex items-center justify-between">
                <div className="mr-3">{<Logo />}</div>
                {typeof siteMetadata.headerTitle === 'string' ? (
                  <div className="hidden h-6 text-2xl font-semibold sm:block">
                    {siteMetadata.headerTitle}
                  </div>
                ) : (
                  siteMetadata.headerTitle
                )}
              </div>
            </Link>
          </div> */}
        <div className="flex items-center font-rs text-base leading-5">
          <div className="hidden sm:block">
            <ul className="nav">
              {headerNavLinks.map((link) => (
                <li key={link.title}>
                  <Link
                    href={link.href}
                    className={
                      currentRoute === link.href
                        ? activeNavLinkClassNames
                        : nonActiveNavLinkClassNames
                    }
                  >
                    {link.title}
                  </Link>
                </li>
              ))}
              <span>|</span>
            </ul>
          </div>
          <ThemeSwitch />
          <MobileNav />
        </div>
      </header>
      <div className="mx-auto flex h-screen flex-col justify-between justify-self-center xl:max-w-4xl">
        <main className="mb-auto">{children}</main>
        <Footer />
      </div>
    </SectionContainer>
  )
}

export default LayoutWrapper
