import Script from 'next/script'

const DifyScript = () => {
  return (
    <>
      <Script
        id="dify-chatbot-config"
        strategy="afterInteractive"
        dangerouslySetInnerHTML={{
          __html: `
                        window.difyChatbotConfig = {
                            token: '1vyqhA009GOZeG1k',
                            baseUrl: 'https://ai.riino.site'
                        };
                    `,
        }}
      />
      <Script
        src="https://ai.riino.site/embed.min.js"
        id="1vyqhA009GOZeG1k"
        strategy="afterInteractive"
        defer
      />
      <style jsx global>{`
        #dify-chatbot-bubble-button {
          background-color: #1c64f2 !important;
        }
      `}</style>
    </>
  )
}

export default DifyScript
