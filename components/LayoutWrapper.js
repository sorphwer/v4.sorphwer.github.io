import headerNavLinks from '@/data/headerNavLinks'
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
      {/* No `justify-self-center`: modern Chromium applies justify-self to block boxes,
          which turned this wrapper into a shrink-to-fit box sized by its content. */}
      <div className="mx-auto flex h-screen flex-col justify-between lg:max-w-5xl xl:max-w-6xl">
        <main className="mb-auto">{children}</main>
        <Footer />
      </div>
    </SectionContainer>
  )
}

export default LayoutWrapper
