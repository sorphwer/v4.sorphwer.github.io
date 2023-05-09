import Link from './Link'
import siteMetadata from '@/data/siteMetadata'
import SocialIcon from '@/components/social-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import Image from 'next/image'

export default function Footer() {
  return (
    <footer>
      <div className="mt-16 flex flex-col items-center font-rs">
        <div className="mb-3 flex space-x-4">
          {/* <SocialIcon kind="mail" href={`mailto:${siteMetadata.email}`} size="6" /> */}
          {/* <SocialIcon kind="github" href={siteMetadata.github} size="6" /> */}
          <Image
            className="brightness-0 filter  dark:brightness-200 dark:filter"
            src="/static/images/logo_Nest.png"
            width={30}
            height={30}
            alt="Picture of the author"
          />
          {/* <SocialIcon kind="facebook" href={siteMetadata.facebook} size="6" />
          <SocialIcon kind="youtube" href={siteMetadata.youtube} size="6" />
          <SocialIcon kind="linkedin" href={siteMetadata.linkedin} size="6" />
          <SocialIcon kind="twitter" href={siteMetadata.twitter} size="6" /> */}
        </div>
        <div className="mb-2 flex space-x-2 text-sm text-gray-500 dark:text-gray-400">
          <div>{`©2012 - ${new Date().getFullYear()} `}</div>
          <div>
            <Link className="hover:text-primary-light" href="/">
              {siteMetadata.title + ' All Rights Reserved.'}
            </Link>
          </div>
        </div>
        <div className="mb-2 flex space-x-2 text-sm text-gray-500 dark:text-gray-400">
          <div className="text-center">
            Nest of Etamine Study - 10th Anniversary <br /> 2012-2022
          </div>
        </div>
      </div>
    </footer>
  )
}
