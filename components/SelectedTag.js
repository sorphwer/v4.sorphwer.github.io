import Link from 'next/link'
import kebabCase from '@/lib/utils/kebabCase'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import Image from 'next/image'

const SelectedTag = ({ text }) => {
  if (kebabCase(text) === 'notion') {
    return (
      <Link href={`/tags/${kebabCase(text)}`}>
        <a
          className="mt-1 mr-3 rounded border-2 border-solid   border-primary-500 bg-gray-300 px-2
            text-sm
          font-medium uppercase
          text-primary-500 transition
          duration-500 ease-out
           dark:border-primary-400
          dark:bg-gray-500 dark:text-primary-400"
        >
          <Image
            className="brightness-0 filter  dark:brightness-200 dark:filter"
            src="/static/images/notion.svg"
            width={14}
            height={14}
            alt="Notion Blog"
          />

          {/* <FontAwesomeIcon icon="tags" className='dark:text-gray-300 text-black ' /> */}
          {' ' + text.split(' ').join('-')}
        </a>
      </Link>
    )
  } else if (kebabCase(text) === 'mdx') {
    return (
      <Link href={`/tags/${kebabCase(text)}`}>
        <a
          className="mt-1 mr-3 rounded border-2 border-solid  border-primary-500 bg-white p-0 text-sm
            font-medium uppercase text-primary-500
          transition duration-500
           ease-out 
         dark:border-primary-400
           dark:text-primary-400"
        >
          <Image src="/static/images/mdx.png" width={34} height={14} alt="mdx" />

          {/* <FontAwesomeIcon icon="tags" className='dark:text-gray-300 text-black ' /> */}
          {/* {' ' + text.split(' ').join('-')} */}
        </a>
      </Link>
    )
  } else {
    return (
      <Link href={`/tags/${kebabCase(text)}`}>
        <a
          className="mt-1 mr-3 rounded border-2 border-solid  border-primary-500 bg-gray-300 px-2 text-sm
           font-medium uppercase
          text-primary-500 transition
          duration-500 ease-out
   dark:border-primary-400
          dark:bg-gray-500 dark:text-primary-400"
        >
          <FontAwesomeIcon icon="tags" className="text-black dark:text-gray-300 " />
          {' ' + text.split(' ').join('-')}
        </a>
      </Link>
    )
  }
}

export default SelectedTag
