;(() => {
  var exports = {}
  exports.id = 888
  exports.ids = [888]
  exports.modules = {
    /***/ 1401: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
      'use strict'
      /* harmony export */ __webpack_require__.d(__webpack_exports__, {
        /* harmony export */ R: () => /* binding */ ClientReload,
        /* harmony export */
      })
      /* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(6689)
      /* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default =
        /*#__PURE__*/ __webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__)
      /* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(1853)
      /* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_1___default =
        /*#__PURE__*/ __webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_1__)

      /**
       * Client-side complement to next-remote-watch
       * Re-triggers getStaticProps when watched mdx files change
       *
       */ const ClientReload = () => {
        // Exclude socket.io from prod bundle
        ;(0, react__WEBPACK_IMPORTED_MODULE_0__.useEffect)(() => {
          Promise.resolve(/* import() */)
            .then(__webpack_require__.bind(__webpack_require__, 4612))
            .then((module) => {
              const socket = module.io()
              socket.on('reload', (data) => {
                next_router__WEBPACK_IMPORTED_MODULE_1___default().replace(
                  next_router__WEBPACK_IMPORTED_MODULE_1___default().asPath,
                  undefined,
                  {
                    scroll: false,
                  }
                )
              })
            })
        }, [])
        return null
      }

      /***/
    },

    /***/ 7661: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
      'use strict'
      /* harmony export */ __webpack_require__.d(__webpack_exports__, {
        /* harmony export */ Z: () => __WEBPACK_DEFAULT_EXPORT__,
        /* harmony export */
      })
      /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ =
        __webpack_require__(997)
      /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default =
        /*#__PURE__*/ __webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__)
      /* harmony import */ var next_image__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(5675)

      // eslint-disable-next-line jsx-a11y/alt-text
      const Image = ({ ...rest }) =>
        /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
          next_image__WEBPACK_IMPORTED_MODULE_1__['default'],
          {
            ...rest,
          }
        )
      /* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = Image

      /***/
    },

    /***/ 7873: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
      'use strict'

      // EXPORTS
      __webpack_require__.d(__webpack_exports__, {
        Z: () => /* binding */ components_LayoutWrapper,
      })

      // EXTERNAL MODULE: external "react/jsx-runtime"
      var jsx_runtime_ = __webpack_require__(997)
      // EXTERNAL MODULE: ./data/siteMetadata.js
      var siteMetadata = __webpack_require__(1576)
      var siteMetadata_default = /*#__PURE__*/ __webpack_require__.n(siteMetadata) // CONCATENATED MODULE: ./data/headerNavLinks.js
      const headerNavLinks = [
        {
          href: '/',
          title: 'Home',
        },
        {
          href: '/blog',
          title: 'archive',
        },
        {
          href: '/tags',
          title: 'Tags',
        },
        // { href: '/projects', title: 'Projects' },
        {
          href: '/profile',
          title: 'Profile',
        },
        {
          href: '/about',
          title: 'About',
        },
        {
          href: 'https://jupyter.riino.site/lab/index.html',
          title: 'Jupyter\u2197',
        },
      ]
      /* harmony default export */ const data_headerNavLinks = headerNavLinks

      // EXTERNAL MODULE: ./components/Image.js
      var Image = __webpack_require__(7661)
      // EXTERNAL MODULE: ./components/Link.js
      var Link = __webpack_require__(7233)
      // EXTERNAL MODULE: ./components/SectionContainer.js
      var SectionContainer = __webpack_require__(890)
      // EXTERNAL MODULE: ./components/social-icons/index.js + 7 modules
      var social_icons = __webpack_require__(9159)
      // EXTERNAL MODULE: external "@fortawesome/react-fontawesome"
      var react_fontawesome_ = __webpack_require__(7197)
      // EXTERNAL MODULE: ./node_modules/next/image.js
      var next_image = __webpack_require__(5675) // CONCATENATED MODULE: ./components/Footer.js
      function Footer() {
        return /*#__PURE__*/ jsx_runtime_.jsx('footer', {
          children: /*#__PURE__*/ (0, jsx_runtime_.jsxs)('div', {
            className: 'mt-16 flex flex-col items-center font-rs',
            children: [
              /*#__PURE__*/ jsx_runtime_.jsx('div', {
                className: 'mb-3 flex space-x-4',
                children: /*#__PURE__*/ jsx_runtime_.jsx(next_image['default'], {
                  className: 'brightness-0 filter dark:brightness-200 dark:filter',
                  src: '/static/images/logo_Nest.png',
                  width: 30,
                  height: 30,
                  alt: 'Picture of the author',
                }),
              }),
              /*#__PURE__*/ (0, jsx_runtime_.jsxs)('div', {
                className: 'mb-2 flex space-x-2 text-sm text-gray-500 dark:text-gray-400',
                children: [
                  /*#__PURE__*/ jsx_runtime_.jsx('div', {
                    children: `©2012 - ${new Date().getFullYear()} `,
                  }),
                  /*#__PURE__*/ jsx_runtime_.jsx('div', {
                    children: /*#__PURE__*/ jsx_runtime_.jsx(Link /* default */.Z, {
                      className: 'hover:text-primary-light',
                      href: '/',
                      children: siteMetadata_default().title + ' All Rights Reserved.',
                    }),
                  }),
                ],
              }),
              /*#__PURE__*/ jsx_runtime_.jsx('div', {
                className: 'mb-2 flex space-x-2 text-sm text-gray-500 dark:text-gray-400',
                children: /*#__PURE__*/ (0, jsx_runtime_.jsxs)('div', {
                  className: 'text-center',
                  children: [
                    /*#__PURE__*/ (0, jsx_runtime_.jsxs)('p', {
                      className: 'mb-3 mt-3 text-black dark:text-white',
                      children: [
                        /*#__PURE__*/ jsx_runtime_.jsx(Link /* default */.Z, {
                          className: 'hover:text-primary-light',
                          href: 'https://riino.site/terms_of_use',
                          children: 'Terms of Use',
                        }),
                        ' ',
                        '|',
                        ' ',
                        /*#__PURE__*/ jsx_runtime_.jsx(Link /* default */.Z, {
                          className: 'hover:text-primary-light',
                          href: 'https://riino.site/privacy_statement/',
                          children: 'Privacy Statement',
                        }),
                      ],
                    }),
                    /*#__PURE__*/ jsx_runtime_.jsx('p', {
                      children: 'Designed, Developed,and Deployed by Riino',
                    }),
                    /*#__PURE__*/ (0, jsx_runtime_.jsxs)('p', {
                      children: [
                        'Nest of Etamine Study - 10th Anniversary ',
                        /*#__PURE__*/ jsx_runtime_.jsx('br', {}),
                        ' 2012-2022',
                      ],
                    }),
                  ],
                }),
              }),
            ],
          }),
        })
      }

      // EXTERNAL MODULE: external "react"
      var external_react_ = __webpack_require__(6689) // CONCATENATED MODULE: ./components/MobileNav.js
      const MobileNav = () => {
        const { 0: navShow, 1: setNavShow } = (0, external_react_.useState)(false)
        const onToggleNav = () => {
          setNavShow((status) => {
            if (status) {
              document.body.style.overflow = 'auto'
            } else {
              // Prevent scrolling
              document.body.style.overflow = 'hidden'
            }
            return !status
          })
        }
        return /*#__PURE__*/ (0, jsx_runtime_.jsxs)('div', {
          className: 'sm:hidden',
          children: [
            /*#__PURE__*/ jsx_runtime_.jsx('button', {
              type: 'button',
              className: 'ml-1 mr-1 h-8 w-8 rounded py-1',
              'aria-label': 'Toggle Menu',
              onClick: onToggleNav,
              children: /*#__PURE__*/ jsx_runtime_.jsx('svg', {
                xmlns: 'http://www.w3.org/2000/svg',
                viewBox: '0 0 20 20',
                fill: 'currentColor',
                className: 'text-gray-900 dark:text-gray-100',
                children: /*#__PURE__*/ jsx_runtime_.jsx('path', {
                  fillRule: 'evenodd',
                  d: 'M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z',
                  clipRule: 'evenodd',
                }),
              }),
            }),
            /*#__PURE__*/ (0, jsx_runtime_.jsxs)('div', {
              className: `fixed top-0 left-0 z-10 h-full w-full transform bg-gray-200 opacity-95 duration-300 ease-in-out dark:bg-gray-800 ${
                navShow ? 'translate-x-0' : 'translate-x-full'
              }`,
              children: [
                /*#__PURE__*/ jsx_runtime_.jsx('div', {
                  className: 'flex justify-end',
                  children: /*#__PURE__*/ jsx_runtime_.jsx('button', {
                    type: 'button',
                    className: 'mr-5 mt-11 h-8 w-8 rounded',
                    'aria-label': 'Toggle Menu',
                    onClick: onToggleNav,
                    children: /*#__PURE__*/ jsx_runtime_.jsx('svg', {
                      xmlns: 'http://www.w3.org/2000/svg',
                      viewBox: '0 0 20 20',
                      fill: 'currentColor',
                      className: 'text-gray-900 dark:text-gray-100',
                      children: /*#__PURE__*/ jsx_runtime_.jsx('path', {
                        fillRule: 'evenodd',
                        d: 'M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z',
                        clipRule: 'evenodd',
                      }),
                    }),
                  }),
                }),
                /*#__PURE__*/ jsx_runtime_.jsx('nav', {
                  className: 'fixed mt-8 h-full',
                  children: data_headerNavLinks.map((link) =>
                    /*#__PURE__*/ jsx_runtime_.jsx(
                      'div',
                      {
                        className: 'px-12 py-4',
                        children: /*#__PURE__*/ jsx_runtime_.jsx(Link /* default */.Z, {
                          href: link.href,
                          className:
                            'text-2xl font-bold tracking-widest text-gray-900 dark:text-gray-100',
                          onClick: onToggleNav,
                          children: link.title,
                        }),
                      },
                      link.title
                    )
                  ),
                }),
              ],
            }),
          ],
        })
      }
      /* harmony default export */ const components_MobileNav = MobileNav

      // EXTERNAL MODULE: external "next-themes"
      var external_next_themes_ = __webpack_require__(1162) // CONCATENATED MODULE: ./components/ThemeSwitch.js
      const ThemeSwitch = () => {
        const { 0: mounted, 1: setMounted } = (0, external_react_.useState)(false)
        const { theme, setTheme, resolvedTheme } = (0, external_next_themes_.useTheme)()
        // When mounted on client, now we can show the UI
        ;(0, external_react_.useEffect)(() => setMounted(true), [])
        return /*#__PURE__*/ jsx_runtime_.jsx('button', {
          'aria-label': 'Toggle Dark Mode',
          type: 'button',
          className: 'ml-1 mr-1 h-8 w-8 rounded p-1 sm:ml-4 ',
          onClick: () => setTheme(theme === 'dark' || resolvedTheme === 'dark' ? 'light' : 'dark'),
          children: /*#__PURE__*/ jsx_runtime_.jsx('svg', {
            xmlns: 'http://www.w3.org/2000/svg',
            viewBox: '0 0 20 20',
            fill: 'currentColor',
            className:
              'text-gray-900 hover:text-primary-light dark:text-gray-100 hover:dark:text-primary-light',
            children:
              mounted && (theme === 'dark' || resolvedTheme === 'dark')
                ? /*#__PURE__*/ jsx_runtime_.jsx('path', {
                    fillRule: 'evenodd',
                    d: 'M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z',
                    clipRule: 'evenodd',
                  })
                : /*#__PURE__*/ jsx_runtime_.jsx('path', {
                    d: 'M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z',
                  }),
          }),
        })
      }
      /* harmony default export */ const components_ThemeSwitch = ThemeSwitch

      // EXTERNAL MODULE: external "next/router"
      var router_ = __webpack_require__(1853) // CONCATENATED MODULE: ./components/LayoutWrapper.js
      // import Logo from '@/data/logo.svg'

      const LayoutWrapper = ({ children }) => {
        const activeNavLinkClassNames = 'active'
        const nonActiveNavLinkClassNames = 'nonActive'
        const currentRoute = (0, router_.useRouter)().pathname
        return /*#__PURE__*/ (0, jsx_runtime_.jsxs)(SectionContainer /* default */.Z, {
          children: [
            /*#__PURE__*/ jsx_runtime_.jsx('header', {
              className: 'mt-10 flex items-center justify-between py-10',
              children: /*#__PURE__*/ (0, jsx_runtime_.jsxs)('div', {
                className: 'flex items-center font-rs text-base leading-5',
                children: [
                  /*#__PURE__*/ jsx_runtime_.jsx('div', {
                    className: 'hidden sm:block',
                    children: /*#__PURE__*/ (0, jsx_runtime_.jsxs)('ul', {
                      className: 'nav',
                      children: [
                        data_headerNavLinks.map((link) =>
                          /*#__PURE__*/ jsx_runtime_.jsx(
                            'li',
                            {
                              children: /*#__PURE__*/ jsx_runtime_.jsx(Link /* default */.Z, {
                                href: link.href,
                                className:
                                  currentRoute === link.href
                                    ? activeNavLinkClassNames
                                    : nonActiveNavLinkClassNames,
                                children: link.title,
                              }),
                            },
                            link.title
                          )
                        ),
                        /*#__PURE__*/ jsx_runtime_.jsx('span', {
                          children: '|',
                        }),
                      ],
                    }),
                  }),
                  /*#__PURE__*/ jsx_runtime_.jsx(components_ThemeSwitch, {}),
                  /*#__PURE__*/ jsx_runtime_.jsx(components_MobileNav, {}),
                ],
              }),
            }),
            /*#__PURE__*/ (0, jsx_runtime_.jsxs)('div', {
              className:
                'mx-auto flex h-screen flex-col justify-between justify-self-center lg:max-w-5xl xl:max-w-6xl',
              children: [
                /*#__PURE__*/ jsx_runtime_.jsx('main', {
                  className: 'mb-auto',
                  children: children,
                }),
                /*#__PURE__*/ jsx_runtime_.jsx(Footer, {}),
              ],
            }),
          ],
        })
      }
      /* harmony default export */ const components_LayoutWrapper = LayoutWrapper

      /***/
    },

    /***/ 890: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
      'use strict'
      /* harmony export */ __webpack_require__.d(__webpack_exports__, {
        /* harmony export */ Z: () => /* binding */ SectionContainer,
        /* harmony export */
      })
      /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ =
        __webpack_require__(997)
      /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default =
        /*#__PURE__*/ __webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__)

      function SectionContainer({ children }) {
        return /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('div', {
          className: 'mx-auto max-w-3xl px-4 sm:px-6 xl:max-w-6xl xl:px-0',
          children: children,
        })
      }

      /***/
    },

    /***/ 9213: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
      'use strict'

      // EXPORTS
      __webpack_require__.d(__webpack_exports__, {
        Z: () => /* binding */ analytics,
      })

      // EXTERNAL MODULE: external "react/jsx-runtime"
      var jsx_runtime_ = __webpack_require__(997) // CONCATENATED MODULE: external "next/script"
      const script_namespaceObject = require('next/script')
      var script_default = /*#__PURE__*/ __webpack_require__.n(script_namespaceObject)
      // EXTERNAL MODULE: ./data/siteMetadata.js
      var siteMetadata = __webpack_require__(1576)
      var siteMetadata_default = /*#__PURE__*/ __webpack_require__.n(siteMetadata) // CONCATENATED MODULE: ./components/analytics/GoogleAnalytics.js
      const GA_Script = () => {
        return /*#__PURE__*/ (0, jsx_runtime_.jsxs)(jsx_runtime_.Fragment, {
          children: [
            /*#__PURE__*/ jsx_runtime_.jsx(script_default(), {
              strategy: 'lazyOnload',
              src: `https://www.googletagmanager.com/gtag/js?id=${
                siteMetadata_default().analytics.googleAnalyticsId
              }`,
            }),
            /*#__PURE__*/ jsx_runtime_.jsx(script_default(), {
              strategy: 'lazyOnload',
              id: 'ga-script',
              children: `
              window.dataLayer = window.dataLayer || [];
              function gtag(){dataLayer.push(arguments);}
              gtag('js', new Date());

              gtag('config', '${siteMetadata_default().analytics.googleAnalyticsId}');
        `,
            }),
          ],
        })
      }
      /* harmony default export */ const GoogleAnalytics = GA_Script
      // https://developers.google.com/analytics/devguides/collection/gtagjs/events
      const logEvent = (action, category, label, value) => {
        var ref
        ;(ref = window.gtag) === null || ref === void 0
          ? void 0
          : ref.call(window, 'event', action, {
              event_category: category,
              event_label: label,
              value: value,
            })
      } // CONCATENATED MODULE: ./components/analytics/Plausible.js

      const PlausibleScript = () => {
        return /*#__PURE__*/ (0, jsx_runtime_.jsxs)(jsx_runtime_.Fragment, {
          children: [
            /*#__PURE__*/ jsx_runtime_.jsx(script_default(), {
              strategy: 'lazyOnload',
              'data-domain': siteMetadata_default().analytics.plausibleDataDomain,
              src: 'https://plausible.io/js/plausible.js',
            }),
            /*#__PURE__*/ jsx_runtime_.jsx(script_default(), {
              strategy: 'lazyOnload',
              id: 'plausible-script',
              children: `
            window.plausible = window.plausible || function() { (window.plausible.q = window.plausible.q || []).push(arguments) }
        `,
            }),
          ],
        })
      }
      /* harmony default export */ const Plausible = PlausibleScript
      // https://plausible.io/docs/custom-event-goals
      const Plausible_logEvent = (eventName, ...rest) => {
        var ref
        return (ref = window.plausible) === null || ref === void 0
          ? void 0
          : ref.call(window, eventName, ...rest)
      } // CONCATENATED MODULE: ./components/analytics/SimpleAnalytics.js

      const SimpleAnalyticsScript = () => {
        return /*#__PURE__*/ (0, jsx_runtime_.jsxs)(jsx_runtime_.Fragment, {
          children: [
            /*#__PURE__*/ jsx_runtime_.jsx(script_default(), {
              strategy: 'lazyOnload',
              id: 'sa-script',
              children: `
            window.sa_event=window.sa_event||function(){var a=[].slice.call(arguments);window.sa_event.q?window.sa_event.q.push(a):window.sa_event.q=[a]};
        `,
            }),
            /*#__PURE__*/ jsx_runtime_.jsx(script_default(), {
              strategy: 'lazyOnload',
              src: 'https://scripts.simpleanalyticscdn.com/latest.js',
            }),
          ],
        })
      }
      // https://docs.simpleanalytics.com/events
      const SimpleAnalytics_logEvent = (eventName, callback) => {
        if (callback) {
          var ref
          return (ref = window.sa_event) === null || ref === void 0
            ? void 0
            : ref.call(window, eventName, callback)
        } else {
          var ref1
          return (ref1 = window.sa_event) === null || ref1 === void 0
            ? void 0
            : ref1.call(window, eventName)
        }
      }
      /* harmony default export */ const SimpleAnalytics = SimpleAnalyticsScript // CONCATENATED MODULE: ./components/analytics/Umami.js

      const UmamiScript = () => {
        return /*#__PURE__*/ jsx_runtime_.jsx(jsx_runtime_.Fragment, {
          children: /*#__PURE__*/ jsx_runtime_.jsx(script_default(), {
            async: true,
            defer: true,
            'data-website-id': siteMetadata_default().analytics.umamiWebsiteId,
            src: 'https://umami.example.com/umami.js', // Replace with your umami instance
          }),
        })
      }
      /* harmony default export */ const Umami = UmamiScript // CONCATENATED MODULE: ./components/analytics/Posthog.js

      const PosthogScript = () => {
        return /*#__PURE__*/ jsx_runtime_.jsx(jsx_runtime_.Fragment, {
          children: /*#__PURE__*/ jsx_runtime_.jsx(script_default(), {
            strategy: 'lazyOnload',
            id: 'posthog-script',
            children: `
            !function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement("script")).type="text/javascript",p.async=!0,p.src=s.api_host+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e},u.people.toString=function(){return u.toString(1)+".people (stub)"},o="capture identify alias people.set people.set_once set_config register register_once unregister opt_out_capturing has_opted_out_capturing opt_in_capturing reset isFeatureEnabled onFeatureFlags".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);
            posthog.init('${
              siteMetadata_default().analytics.posthogAnalyticsId
            }',{api_host:'https://app.posthog.com'})
        `,
          }),
        })
      }
      /* harmony default export */ const Posthog = PosthogScript // CONCATENATED MODULE: external "styled-jsx/style"

      const style_namespaceObject = require('styled-jsx/style')
      var style_default = /*#__PURE__*/ __webpack_require__.n(style_namespaceObject) // CONCATENATED MODULE: ./components/analytics/dify.js
      const DifyScript = () => {
        return /*#__PURE__*/ (0, jsx_runtime_.jsxs)(jsx_runtime_.Fragment, {
          children: [
            /*#__PURE__*/ jsx_runtime_.jsx(script_default(), {
              id: 'dify-chatbot-config',
              strategy: 'afterInteractive',
              dangerouslySetInnerHTML: {
                __html: `
                        window.difyChatbotConfig = {
                            token: '1vyqhA009GOZeG1k',
                            baseUrl: 'https://ai.riino.site',
                                containerProps: {
                                className: 'dify-chatbot-bubble-button-custom',
                                },
                        };
                    `,
              },
            }),
            /*#__PURE__*/ jsx_runtime_.jsx(script_default(), {
              src: 'https://ai.riino.site/embed.min.js',
              id: '1vyqhA009GOZeG1k',
              strategy: 'afterInteractive',
              /*#__PURE__*/ defer: true,
            }),
            jsx_runtime_.jsx(style_default(), {
              id: '9b9407d99706a8d5',
              children: '#dify-chatbot-bubble-button{background-color:#1c64f2!important}',
            }),
          ],
        })
      }
      /* harmony default export */ const dify = DifyScript // CONCATENATED MODULE: ./components/analytics/index.js

      const isProduction = 'production' === 'production'
      const Analytics = () => {
        return /*#__PURE__*/ (0, jsx_runtime_.jsxs)(jsx_runtime_.Fragment, {
          children: [
            isProduction &&
              siteMetadata_default().analytics.plausibleDataDomain &&
              /*#__PURE__*/ jsx_runtime_.jsx(Plausible, {}),
            isProduction &&
              siteMetadata_default().analytics.simpleAnalytics &&
              /*#__PURE__*/ jsx_runtime_.jsx(SimpleAnalytics, {}),
            isProduction &&
              siteMetadata_default().analytics.umamiWebsiteId &&
              /*#__PURE__*/ jsx_runtime_.jsx(Umami, {}),
            siteMetadata_default().analytics.googleAnalyticsId &&
              /*#__PURE__*/ jsx_runtime_.jsx(GoogleAnalytics, {}),
            isProduction &&
              siteMetadata_default().analytics.posthogAnalyticsId &&
              /*#__PURE__*/ jsx_runtime_.jsx(Posthog, {}),
            /*#__PURE__*/ jsx_runtime_.jsx(dify, {}),
          ],
        })
      }
      /* harmony default export */ const analytics = Analytics

      /***/
    },

    /***/ 8484: /***/ (module, __webpack_exports__, __webpack_require__) => {
      'use strict'
      __webpack_require__.a(
        module,
        async (__webpack_handle_async_dependencies__, __webpack_async_result__) => {
          try {
            __webpack_require__.r(__webpack_exports__)
            /* harmony export */ __webpack_require__.d(__webpack_exports__, {
              /* harmony export */ default: () => /* binding */ App,
              /* harmony export */
            })
            /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ =
              __webpack_require__(997)
            /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default =
              /*#__PURE__*/ __webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__)
            /* harmony import */ var _fortawesome_fontawesome_svg_core_styles_css__WEBPACK_IMPORTED_MODULE_1__ =
              __webpack_require__(5800)
            /* harmony import */ var _fortawesome_fontawesome_svg_core_styles_css__WEBPACK_IMPORTED_MODULE_1___default =
              /*#__PURE__*/ __webpack_require__.n(
                _fortawesome_fontawesome_svg_core_styles_css__WEBPACK_IMPORTED_MODULE_1__
              )
            /* harmony import */ var next_themes__WEBPACK_IMPORTED_MODULE_2__ =
              __webpack_require__(1162)
            /* harmony import */ var next_themes__WEBPACK_IMPORTED_MODULE_2___default =
              /*#__PURE__*/ __webpack_require__.n(next_themes__WEBPACK_IMPORTED_MODULE_2__)
            /* harmony import */ var next_head__WEBPACK_IMPORTED_MODULE_3__ =
              __webpack_require__(968)
            /* harmony import */ var next_head__WEBPACK_IMPORTED_MODULE_3___default =
              /*#__PURE__*/ __webpack_require__.n(next_head__WEBPACK_IMPORTED_MODULE_3__)
            /* harmony import */ var _data_siteMetadata__WEBPACK_IMPORTED_MODULE_4__ =
              __webpack_require__(1576)
            /* harmony import */ var _data_siteMetadata__WEBPACK_IMPORTED_MODULE_4___default =
              /*#__PURE__*/ __webpack_require__.n(_data_siteMetadata__WEBPACK_IMPORTED_MODULE_4__)
            /* harmony import */ var _components_analytics__WEBPACK_IMPORTED_MODULE_5__ =
              __webpack_require__(9213)
            /* harmony import */ var _components_LayoutWrapper__WEBPACK_IMPORTED_MODULE_6__ =
              __webpack_require__(7873)
            /* harmony import */ var _components_ClientReload__WEBPACK_IMPORTED_MODULE_7__ =
              __webpack_require__(1401)
            /* harmony import */ var _auth0_nextjs_auth0_client__WEBPACK_IMPORTED_MODULE_8__ =
              __webpack_require__(6153)
            /* harmony import */ var _auth0_nextjs_auth0_client__WEBPACK_IMPORTED_MODULE_8___default =
              /*#__PURE__*/ __webpack_require__.n(
                _auth0_nextjs_auth0_client__WEBPACK_IMPORTED_MODULE_8__
              )
            /* harmony import */ var _fortawesome_fontawesome_svg_core__WEBPACK_IMPORTED_MODULE_9__ =
              __webpack_require__(86)
            /* harmony import */ var _fortawesome_free_solid_svg_icons__WEBPACK_IMPORTED_MODULE_10__ =
              __webpack_require__(4563)
            var __webpack_async_dependencies__ = __webpack_handle_async_dependencies__([
              _fortawesome_fontawesome_svg_core__WEBPACK_IMPORTED_MODULE_9__,
              _fortawesome_free_solid_svg_icons__WEBPACK_IMPORTED_MODULE_10__,
            ])
            ;[
              _fortawesome_fontawesome_svg_core__WEBPACK_IMPORTED_MODULE_9__,
              _fortawesome_free_solid_svg_icons__WEBPACK_IMPORTED_MODULE_10__,
            ] = __webpack_async_dependencies__.then
              ? (await __webpack_async_dependencies__)()
              : __webpack_async_dependencies__

            _fortawesome_fontawesome_svg_core__WEBPACK_IMPORTED_MODULE_9__.config.autoAddCss = false
            _fortawesome_fontawesome_svg_core__WEBPACK_IMPORTED_MODULE_9__.library.add(
              _fortawesome_free_solid_svg_icons__WEBPACK_IMPORTED_MODULE_10__.faTags,
              _fortawesome_free_solid_svg_icons__WEBPACK_IMPORTED_MODULE_10__.faEdit,
              _fortawesome_free_solid_svg_icons__WEBPACK_IMPORTED_MODULE_10__.faSun,
              _fortawesome_free_solid_svg_icons__WEBPACK_IMPORTED_MODULE_10__.faSnowflake,
              _fortawesome_free_solid_svg_icons__WEBPACK_IMPORTED_MODULE_10__.faFan,
              _fortawesome_free_solid_svg_icons__WEBPACK_IMPORTED_MODULE_10__.faLeaf
            )
            const isDevelopment = 'production' === 'development'
            const isSocket = process.env.SOCKET
            function App({ Component, pageProps }) {
              return /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
                next_themes__WEBPACK_IMPORTED_MODULE_2__.ThemeProvider,
                {
                  attribute: 'class',
                  defaultTheme: _data_siteMetadata__WEBPACK_IMPORTED_MODULE_4___default().theme,
                  children: [
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                      next_head__WEBPACK_IMPORTED_MODULE_3___default(),
                      {
                        children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                          'meta',
                          {
                            content: 'width=device-width, initial-scale=1',
                            name: 'viewport',
                          }
                        ),
                      }
                    ),
                    isDevelopment &&
                      isSocket &&
                      /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                        _components_ClientReload__WEBPACK_IMPORTED_MODULE_7__ /* .ClientReload */.R,
                        {}
                      ),
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                      _components_analytics__WEBPACK_IMPORTED_MODULE_5__ /* ["default"] */.Z,
                      {}
                    ),
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                      _auth0_nextjs_auth0_client__WEBPACK_IMPORTED_MODULE_8__.UserProvider,
                      {
                        children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                          _components_LayoutWrapper__WEBPACK_IMPORTED_MODULE_6__ /* ["default"] */.Z,
                          {
                            children:
                              /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                Component,
                                {
                                  ...pageProps,
                                }
                              ),
                          }
                        ),
                      }
                    ),
                  ],
                }
              )
            }

            __webpack_async_result__()
          } catch (e) {
            __webpack_async_result__(e)
          }
        }
      )

      /***/
    },

    /***/ 5800: /***/ () => {
      /***/
    },

    /***/ 6153: /***/ (module) => {
      'use strict'
      module.exports = require('@auth0/nextjs-auth0/client')

      /***/
    },

    /***/ 7197: /***/ (module) => {
      'use strict'
      module.exports = require('@fortawesome/react-fontawesome')

      /***/
    },

    /***/ 1162: /***/ (module) => {
      'use strict'
      module.exports = require('next-themes')

      /***/
    },

    /***/ 562: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/server/denormalize-page-path.js')

      /***/
    },

    /***/ 4957: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/head.js')

      /***/
    },

    /***/ 4014: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/i18n/normalize-locale-path.js')

      /***/
    },

    /***/ 744: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/image-config-context.js')

      /***/
    },

    /***/ 5843: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/image-config.js')

      /***/
    },

    /***/ 8524: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/is-plain-object.js')

      /***/
    },

    /***/ 8020: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/mitt.js')

      /***/
    },

    /***/ 4964: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/router-context.js')

      /***/
    },

    /***/ 3938: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/router/utils/format-url.js')

      /***/
    },

    /***/ 9565: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/router/utils/get-asset-path-from-route.js')

      /***/
    },

    /***/ 1428: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/router/utils/is-dynamic.js')

      /***/
    },

    /***/ 1292: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/router/utils/parse-relative-url.js')

      /***/
    },

    /***/ 979: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/router/utils/querystring.js')

      /***/
    },

    /***/ 6052: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/router/utils/resolve-rewrites.js')

      /***/
    },

    /***/ 4226: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/router/utils/route-matcher.js')

      /***/
    },

    /***/ 5052: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/router/utils/route-regex.js')

      /***/
    },

    /***/ 4241: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/router/utils/routing-items.js')

      /***/
    },

    /***/ 9232: /***/ (module) => {
      'use strict'
      module.exports = require('next/dist/shared/lib/utils.js')

      /***/
    },

    /***/ 968: /***/ (module) => {
      'use strict'
      module.exports = require('next/head')

      /***/
    },

    /***/ 1853: /***/ (module) => {
      'use strict'
      module.exports = require('next/router')

      /***/
    },

    /***/ 6689: /***/ (module) => {
      'use strict'
      module.exports = require('react')

      /***/
    },

    /***/ 997: /***/ (module) => {
      'use strict'
      module.exports = require('react/jsx-runtime')

      /***/
    },

    /***/ 86: /***/ (module) => {
      'use strict'
      module.exports = import('@fortawesome/fontawesome-svg-core')

      /***/
    },

    /***/ 4563: /***/ (module) => {
      'use strict'
      module.exports = import('@fortawesome/free-solid-svg-icons')

      /***/
    },

    /***/ 4612: /***/ (module) => {
      'use strict'
      module.exports = import('socket.io-client')

      /***/
    },
  }
  // load runtime
  var __webpack_require__ = require('../webpack-runtime.js')
  __webpack_require__.C(exports)
  var __webpack_exec__ = (moduleId) => __webpack_require__((__webpack_require__.s = moduleId))
  var __webpack_exports__ = __webpack_require__.X(0, [895, 664, 675, 776, 159], () =>
    __webpack_exec__(8484)
  )
  module.exports = __webpack_exports__
})()
