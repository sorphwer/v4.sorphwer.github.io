import Link from 'next/link'
import kebabCase from '@/lib/utils/kebabCase'
// riinosite3 tag design
const Tag = ({ text }) => {
  return (
    <Link href={`/tags/${kebabCase(text)}`}>
      <a
        className="mt-1 mr-3 rounded border-2 border-solid border-black px-2 text-sm font-medium uppercase
      text-black transition duration-500
      ease-out hover:border-primary-light
      hover:bg-black hover:text-white dark:border-white 
      dark:text-white dark:hover:border-primary-dark
      hover:dark:bg-white dark:hover:text-black"
      >
        {text.split(' ').join('-')}
      </a>
    </Link>
  )
}

export default Tag
