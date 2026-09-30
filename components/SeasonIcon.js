import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'

// The blog's date marker, one icon per quarter of the year:
// Jan–Mar fan, Apr–Jun sun, Jul–Sep leaf, Oct–Dec snowflake.
const SEASONS = [
  { icon: 'fan', className: 'text-pink-300' },
  { icon: 'sun', className: 'text-amber-300' },
  { icon: 'leaf', className: 'text-green-300' },
  { icon: 'snowflake', className: 'text-stone-300' },
]

export default function SeasonIcon({ date, className = '' }) {
  const season = SEASONS[Math.floor(new Date(date).getMonth() / 3)]
  return <FontAwesomeIcon icon={season.icon} className={`${season.className} ${className}`} />
}
