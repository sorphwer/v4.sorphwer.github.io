exports.id = 751
exports.ids = [751]
exports.modules = {
  /***/ 9618: /***/ (module, __unused_webpack_exports, __webpack_require__) => {
    var map = {
      './AuthorLayout': 4856,
      './AuthorLayout.js': 4856,
      './ListLayout': 6055,
      './ListLayout.js': 6055,
      './PostLayout': 5067,
      './PostLayout.js': 5067,
      './PostSimple': 3168,
      './PostSimple.js': 3168,
    }

    function webpackContext(req) {
      var id = webpackContextResolve(req)
      return __webpack_require__(id)
    }
    function webpackContextResolve(req) {
      if (!__webpack_require__.o(map, req)) {
        var e = new Error("Cannot find module '" + req + "'")
        e.code = 'MODULE_NOT_FOUND'
        throw e
      }
      return map[req]
    }
    webpackContext.keys = function webpackContextKeys() {
      return Object.keys(map)
    }
    webpackContext.resolve = webpackContextResolve
    module.exports = webpackContext
    webpackContext.id = 9618

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

  /***/ 7751: /***/ (module, __webpack_exports__, __webpack_require__) => {
    'use strict'
    __webpack_require__.a(
      module,
      async (__webpack_handle_async_dependencies__, __webpack_async_result__) => {
        try {
          /* harmony export */ __webpack_require__.d(__webpack_exports__, {
            /* harmony export */ J: () => /* binding */ MDXLayoutRenderer,
            /* harmony export */
          })
          /* unused harmony export MDXComponents */
          /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ =
            __webpack_require__(997)
          /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default =
            /*#__PURE__*/ __webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__)
          /* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(6689)
          /* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default =
            /*#__PURE__*/ __webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__)
          /* harmony import */ var mdx_bundler_client__WEBPACK_IMPORTED_MODULE_2__ =
            __webpack_require__(1228)
          /* harmony import */ var mdx_bundler_client__WEBPACK_IMPORTED_MODULE_2___default =
            /*#__PURE__*/ __webpack_require__.n(mdx_bundler_client__WEBPACK_IMPORTED_MODULE_2__)
          /* harmony import */ var _Image__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(7661)
          /* harmony import */ var _Link__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(7233)
          /* harmony import */ var _TOCInline__WEBPACK_IMPORTED_MODULE_5__ =
            __webpack_require__(3073)
          /* harmony import */ var _Pre__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(6539)
          /* harmony import */ var _NewsletterForm__WEBPACK_IMPORTED_MODULE_7__ =
            __webpack_require__(7726)
          /* harmony import */ var react_notion_x__WEBPACK_IMPORTED_MODULE_8__ =
            __webpack_require__(5574)
          /* harmony import */ var notion_utils__WEBPACK_IMPORTED_MODULE_9__ =
            __webpack_require__(8751)
          /* harmony import */ var next_dynamic__WEBPACK_IMPORTED_MODULE_10__ =
            __webpack_require__(5152)
          var __webpack_async_dependencies__ = __webpack_handle_async_dependencies__([
            react_notion_x__WEBPACK_IMPORTED_MODULE_8__,
            notion_utils__WEBPACK_IMPORTED_MODULE_9__,
          ])
          ;[
            react_notion_x__WEBPACK_IMPORTED_MODULE_8__,
            notion_utils__WEBPACK_IMPORTED_MODULE_9__,
          ] = __webpack_async_dependencies__.then
            ? (await __webpack_async_dependencies__)()
            : __webpack_async_dependencies__

          /* eslint-disable react/display-name */

          const Code = (0, next_dynamic__WEBPACK_IMPORTED_MODULE_10__['default'])(
            () =>
              Promise.all(/* import() */ [__webpack_require__.e(830), __webpack_require__.e(794)])
                .then(__webpack_require__.bind(__webpack_require__, 1794))
                .then((m) => m.Code),
            {
              loadableGenerated: {
                modules: [
                  '../components/MDXComponents.js -> ' + 'react-notion-x/build/third-party/code',
                ],
              },
            }
          )
          const Collection = (0, next_dynamic__WEBPACK_IMPORTED_MODULE_10__['default'])(
            () =>
              Promise.all(/* import() */ [__webpack_require__.e(830), __webpack_require__.e(635)])
                .then(__webpack_require__.bind(__webpack_require__, 635))
                .then((m) => m.Collection),
            {
              loadableGenerated: {
                modules: [
                  '../components/MDXComponents.js -> ' +
                    'react-notion-x/build/third-party/collection',
                ],
              },
            }
          )
          const Equation = (0, next_dynamic__WEBPACK_IMPORTED_MODULE_10__['default'])(
            () =>
              Promise.all(/* import() */ [__webpack_require__.e(830), __webpack_require__.e(233)])
                .then(__webpack_require__.bind(__webpack_require__, 233))
                .then((m) => m.Equation),
            {
              loadableGenerated: {
                modules: [
                  '../components/MDXComponents.js -> ' +
                    'react-notion-x/build/third-party/equation',
                ],
              },
            }
          )
          const Pdf = (0, next_dynamic__WEBPACK_IMPORTED_MODULE_10__['default'])(null, {
            loadableGenerated: {
              modules: [
                '../components/MDXComponents.js -> ' + 'react-notion-x/build/third-party/pdf',
              ],
            },
            ssr: false,
          })
          const Modal = (0, next_dynamic__WEBPACK_IMPORTED_MODULE_10__['default'])(null, {
            loadableGenerated: {
              modules: [
                '../components/MDXComponents.js -> ' + 'react-notion-x/build/third-party/modal',
              ],
            },
            ssr: false,
          })
          const MDXComponents = {
            Image: _Image__WEBPACK_IMPORTED_MODULE_3__ /* ["default"] */.Z,
            TOCInline: _TOCInline__WEBPACK_IMPORTED_MODULE_5__ /* ["default"] */.Z,
            a: _Link__WEBPACK_IMPORTED_MODULE_4__ /* ["default"] */.Z,
            pre: _Pre__WEBPACK_IMPORTED_MODULE_6__ /* ["default"] */.Z,
            BlogNewsletterForm:
              _NewsletterForm__WEBPACK_IMPORTED_MODULE_7__ /* .BlogNewsletterForm */.w,
            wrapper: ({ components, layout, ...rest }) => {
              const Layout = __webpack_require__(9618)(`./${layout}`).default
              return /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(Layout, {
                ...rest,
              })
            },
          }
          const MDXLayoutRenderer = ({ layout, mdxSource, recordMap, ...rest }) => {
            const MDXLayout = (0, react__WEBPACK_IMPORTED_MODULE_1__.useMemo)(
              () => (0, mdx_bundler_client__WEBPACK_IMPORTED_MODULE_2__.getMDXComponent)(mdxSource),
              [mdxSource]
            )
            // const { recordMap,...reset } = rest
            const NotionJsx = recordMap
              ? /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                  react_notion_x__WEBPACK_IMPORTED_MODULE_8__.NotionRenderer,
                  {
                    recordMap: recordMap,
                    fullPage: true,
                    darkMode: true,
                    components: {
                      Code,
                      Collection,
                      Equation,
                      Modal,
                      Pdf,
                    },
                  }
                )
              : /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('span', {
                  className: 'noNotion',
                })
            const NotionTitle = recordMap
              ? (0, notion_utils__WEBPACK_IMPORTED_MODULE_9__.getPageTitle)(recordMap)
              : null
            return /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
              react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.Fragment,
              {
                children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                  MDXLayout,
                  {
                    layout: layout,
                    components: MDXComponents,
                    NotionJsx: NotionJsx,
                    NotionTitle: NotionTitle,
                    ...rest,
                  }
                ),
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

  /***/ 920: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
    'use strict'
    /* harmony export */ __webpack_require__.d(__webpack_exports__, {
      /* harmony export */ Z: () => /* binding */ PageTitle,
      /* harmony export */
    })
    /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ =
      __webpack_require__(997)
    /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default =
      /*#__PURE__*/ __webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__)

    function PageTitle({ children }) {
      return /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('h1', {
        className:
          'text-left font-rs text-4xl font-medium leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:text-5xl md:leading-14',
        children: children,
      })
    }

    /***/
  },

  /***/ 6539: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
    'use strict'
    /* harmony export */ __webpack_require__.d(__webpack_exports__, {
      /* harmony export */ Z: () => __WEBPACK_DEFAULT_EXPORT__,
      /* harmony export */
    })
    /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ =
      __webpack_require__(997)
    /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default =
      /*#__PURE__*/ __webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__)
    /* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(6689)
    /* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default =
      /*#__PURE__*/ __webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__)

    const Pre = (props) => {
      const textInput = (0, react__WEBPACK_IMPORTED_MODULE_1__.useRef)(null)
      const { 0: hovered, 1: setHovered } = (0, react__WEBPACK_IMPORTED_MODULE_1__.useState)(false)
      const { 0: copied, 1: setCopied } = (0, react__WEBPACK_IMPORTED_MODULE_1__.useState)(false)
      const onEnter = () => {
        setHovered(true)
      }
      const onExit = () => {
        setHovered(false)
        setCopied(false)
      }
      const onCopy = () => {
        setCopied(true)
        navigator.clipboard.writeText(textInput.current.textContent)
        setTimeout(() => {
          setCopied(false)
        }, 2000)
      }
      return /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('div', {
        ref: textInput,
        onMouseEnter: onEnter,
        onMouseLeave: onExit,
        className: 'relative',
        children: [
          hovered &&
            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('button', {
              'aria-label': 'Copy code',
              type: 'button',
              className: `absolute right-2 top-2 h-8 w-8 rounded border-2 bg-gray-700 p-1 dark:bg-gray-800 ${
                copied
                  ? 'border-green-400 focus:border-green-400 focus:outline-none'
                  : 'border-gray-300'
              }`,
              onClick: onCopy,
              children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('svg', {
                xmlns: 'http://www.w3.org/2000/svg',
                viewBox: '0 0 24 24',
                stroke: 'currentColor',
                fill: 'none',
                className: copied ? 'text-green-400' : 'text-gray-300',
                children: copied
                  ? /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                      react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.Fragment,
                      {
                        children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                          'path',
                          {
                            strokeLinecap: 'round',
                            strokeLinejoin: 'round',
                            strokeWidth: 2,
                            d: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4',
                          }
                        ),
                      }
                    )
                  : /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                      react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.Fragment,
                      {
                        children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                          'path',
                          {
                            strokeLinecap: 'round',
                            strokeLinejoin: 'round',
                            strokeWidth: 2,
                            d: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2',
                          }
                        ),
                      }
                    ),
              }),
            }),
          /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('pre', {
            children: props.children,
          }),
        ],
      })
    }
    /* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = Pre

    /***/
  },

  /***/ 7175: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
    'use strict'
    /* harmony export */ __webpack_require__.d(__webpack_exports__, {
      /* harmony export */ Z: () => __WEBPACK_DEFAULT_EXPORT__,
      /* harmony export */
    })
    /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ =
      __webpack_require__(997)
    /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default =
      /*#__PURE__*/ __webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__)
    /* harmony import */ var _data_siteMetadata__WEBPACK_IMPORTED_MODULE_1__ =
      __webpack_require__(1576)
    /* harmony import */ var _data_siteMetadata__WEBPACK_IMPORTED_MODULE_1___default =
      /*#__PURE__*/ __webpack_require__.n(_data_siteMetadata__WEBPACK_IMPORTED_MODULE_1__)
    /* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(6689)
    /* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default =
      /*#__PURE__*/ __webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__)

    const ScrollTopAndComment = () => {
      const { 0: show, 1: setShow } = (0, react__WEBPACK_IMPORTED_MODULE_2__.useState)(false)
      ;(0, react__WEBPACK_IMPORTED_MODULE_2__.useEffect)(() => {
        const handleWindowScroll = () => {
          if (window.scrollY > 50) setShow(true)
          else setShow(false)
        }
        window.addEventListener('scroll', handleWindowScroll)
        return () => window.removeEventListener('scroll', handleWindowScroll)
      }, [])
      const handleScrollTop = () => {
        window.scrollTo({
          top: 0,
        })
      }
      const handleScrollToComment = () => {
        document.getElementById('comment').scrollIntoView()
      }
      return /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('div', {
        className: `fixed right-8 bottom-8 hidden flex-col gap-3 ${show ? 'md:flex' : 'md:hidden'}`,
        children: [
          _data_siteMetadata__WEBPACK_IMPORTED_MODULE_1___default().comment.provider &&
            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('button', {
              'aria-label': 'Scroll To Comment',
              type: 'button',
              onClick: handleScrollToComment,
              className:
                'rounded-full bg-gray-200 p-2 text-gray-500 transition-all hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-400 dark:hover:bg-gray-600',
              children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('svg', {
                className: 'h-5 w-5',
                viewBox: '0 0 20 20',
                fill: 'currentColor',
                children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('path', {
                  fillRule: 'evenodd',
                  d: 'M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z',
                  clipRule: 'evenodd',
                }),
              }),
            }),
          /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('button', {
            'aria-label': 'Scroll To Top',
            type: 'button',
            onClick: handleScrollTop,
            className:
              'rounded-full bg-gray-200 p-2 text-gray-500 transition-all hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-400 dark:hover:bg-gray-600',
            children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('svg', {
              className: 'h-5 w-5',
              viewBox: '0 0 20 20',
              fill: 'currentColor',
              children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('path', {
                fillRule: 'evenodd',
                d: 'M3.293 9.707a1 1 0 010-1.414l6-6a1 1 0 011.414 0l6 6a1 1 0 01-1.414 1.414L11 5.414V17a1 1 0 11-2 0V5.414L4.707 9.707a1 1 0 01-1.414 0z',
                clipRule: 'evenodd',
              }),
            }),
          }),
        ],
      })
    }
    /* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = ScrollTopAndComment

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

  /***/ 3073: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
    'use strict'
    /* harmony export */ __webpack_require__.d(__webpack_exports__, {
      /* harmony export */ Z: () => __WEBPACK_DEFAULT_EXPORT__,
      /* harmony export */
    })
    /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ =
      __webpack_require__(997)
    /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default =
      /*#__PURE__*/ __webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__)

    /**
     * @typedef TocHeading
     * @prop {string} value
     * @prop {number} depth
     * @prop {string} url
     */ /**
     * Generates an inline table of contents
     * Exclude titles matching this string (new RegExp('^(' + string + ')$', 'i')).
     * If an array is passed the array gets joined with a pipe (new RegExp('^(' + array.join('|') + ')$', 'i')).
     *
     * @param {{
     *  toc: TocHeading[],
     *  indentDepth?: number,
     *  fromHeading?: number,
     *  toHeading?: number,
     *  asDisclosure?: boolean,
     *  exclude?: string|string[]
     * }} props
     *
     */ const TOCInline = ({
      toc,
      indentDepth = 3,
      fromHeading = 1,
      toHeading = 6,
      asDisclosure = false,
      exclude = '',
    }) => {
      const re = Array.isArray(exclude)
        ? new RegExp('^(' + exclude.join('|') + ')$', 'i')
        : new RegExp('^(' + exclude + ')$', 'i')
      const filteredToc = toc.filter(
        (heading) =>
          heading.depth >= fromHeading && heading.depth <= toHeading && !re.test(heading.value)
      )
      const tocList = /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('ul', {
        children: filteredToc.map((heading) =>
          /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
            'li',
            {
              className: `${heading.depth >= indentDepth && 'ml-6'}`,
              children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('a', {
                href: heading.url,
                children: heading.value,
              }),
            },
            heading.value
          )
        ),
      })
      return /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
        react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.Fragment,
        {
          children: asDisclosure
            ? /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('details', {
                open: true,
                children: [
                  /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('summary', {
                    className: 'ml-6 pt-2 pb-2 text-xl font-bold',
                    children: 'Table of Contents',
                  }),
                  /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('div', {
                    className: 'ml-6',
                    children: tocList,
                  }),
                ],
              })
            : tocList,
        }
      )
    }
    /* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = TOCInline

    /***/
  },

  /***/ 896: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
    'use strict'
    /* harmony export */ __webpack_require__.d(__webpack_exports__, {
      /* harmony export */ Z: () => __WEBPACK_DEFAULT_EXPORT__,
      /* harmony export */
    })
    /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ =
      __webpack_require__(997)
    /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default =
      /*#__PURE__*/ __webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__)
    /* harmony import */ var _data_siteMetadata__WEBPACK_IMPORTED_MODULE_1__ =
      __webpack_require__(1576)
    /* harmony import */ var _data_siteMetadata__WEBPACK_IMPORTED_MODULE_1___default =
      /*#__PURE__*/ __webpack_require__.n(_data_siteMetadata__WEBPACK_IMPORTED_MODULE_1__)
    /* harmony import */ var next_dynamic__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(5152)

    const UtterancesComponent = (0, next_dynamic__WEBPACK_IMPORTED_MODULE_2__['default'])(null, {
      loadableGenerated: {
        modules: ['../components/comments/index.js -> ' + '@/components/comments/Utterances'],
      },
      ssr: false,
    })
    const GiscusComponent = (0, next_dynamic__WEBPACK_IMPORTED_MODULE_2__['default'])(null, {
      loadableGenerated: {
        modules: ['../components/comments/index.js -> ' + '@/components/comments/Giscus'],
      },
      ssr: false,
    })
    const DisqusComponent = (0, next_dynamic__WEBPACK_IMPORTED_MODULE_2__['default'])(null, {
      loadableGenerated: {
        modules: ['../components/comments/index.js -> ' + '@/components/comments/Disqus'],
      },
      ssr: false,
    })
    const Comments = ({ frontMatter }) => {
      const comment =
        _data_siteMetadata__WEBPACK_IMPORTED_MODULE_1___default() === null ||
        _data_siteMetadata__WEBPACK_IMPORTED_MODULE_1___default() === void 0
          ? void 0
          : _data_siteMetadata__WEBPACK_IMPORTED_MODULE_1___default().comment
      if (!comment || Object.keys(comment).length === 0)
        return /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
          react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.Fragment,
          {}
        )
      return /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('div', {
        id: 'comment',
        children: [
          _data_siteMetadata__WEBPACK_IMPORTED_MODULE_1___default().comment &&
            _data_siteMetadata__WEBPACK_IMPORTED_MODULE_1___default().comment.provider ===
              'giscus' &&
            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(GiscusComponent, {}),
          _data_siteMetadata__WEBPACK_IMPORTED_MODULE_1___default().comment &&
            _data_siteMetadata__WEBPACK_IMPORTED_MODULE_1___default().comment.provider ===
              'utterances' &&
            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
              UtterancesComponent,
              {}
            ),
          _data_siteMetadata__WEBPACK_IMPORTED_MODULE_1___default().comment &&
            _data_siteMetadata__WEBPACK_IMPORTED_MODULE_1___default().comment.provider ===
              'disqus' &&
            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(DisqusComponent, {
              frontMatter: frontMatter,
            }),
        ],
      })
    }
    /* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = Comments

    /***/
  },

  /***/ 4856: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
    'use strict'
    __webpack_require__.r(__webpack_exports__)
    /* harmony export */ __webpack_require__.d(__webpack_exports__, {
      /* harmony export */ default: () => /* binding */ AuthorLayout,
      /* harmony export */
    })
    /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ =
      __webpack_require__(997)
    /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default =
      /*#__PURE__*/ __webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__)
    /* harmony import */ var _components_social_icons__WEBPACK_IMPORTED_MODULE_1__ =
      __webpack_require__(9159)
    /* harmony import */ var _components_Image__WEBPACK_IMPORTED_MODULE_2__ =
      __webpack_require__(7661)
    /* harmony import */ var _components_SEO__WEBPACK_IMPORTED_MODULE_3__ =
      __webpack_require__(9831)

    // About page

    function AuthorLayout({ children, NotionJsx, frontMatter }) {
      const { name, avatar, occupation, company, email, twitter, linkedin, github } = frontMatter
      return /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
        react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.Fragment,
        {
          children: [
            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
              _components_SEO__WEBPACK_IMPORTED_MODULE_3__ /* .PageSEO */.TQ,
              {
                title: `About - ${name}`,
                description: `About me - ${name}`,
              }
            ),
            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('div', {
              className: 'divide-y divide-gray-200 dark:divide-gray-700',
              children: /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
                'div',
                {
                  className: 'prose max-w-none pt-8 pb-8 dark:prose-dark xl:col-span-2',
                  children: [children, NotionJsx],
                }
              ),
            }),
          ],
        }
      )
    }

    /***/
  },

  /***/ 5067: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
    'use strict'
    __webpack_require__.r(__webpack_exports__)
    /* harmony export */ __webpack_require__.d(__webpack_exports__, {
      /* harmony export */ default: () => /* binding */ PostLayout,
      /* harmony export */
    })
    /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ =
      __webpack_require__(997)
    /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default =
      /*#__PURE__*/ __webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__)
    /* harmony import */ var _components_Link__WEBPACK_IMPORTED_MODULE_1__ =
      __webpack_require__(7233)
    /* harmony import */ var _components_PageTitle__WEBPACK_IMPORTED_MODULE_2__ =
      __webpack_require__(920)
    /* harmony import */ var _components_SectionContainer__WEBPACK_IMPORTED_MODULE_3__ =
      __webpack_require__(890)
    /* harmony import */ var _components_SEO__WEBPACK_IMPORTED_MODULE_4__ =
      __webpack_require__(9831)
    /* harmony import */ var _components_Image__WEBPACK_IMPORTED_MODULE_5__ =
      __webpack_require__(7661)
    /* harmony import */ var _components_Tag__WEBPACK_IMPORTED_MODULE_6__ =
      __webpack_require__(9019)
    /* harmony import */ var _data_siteMetadata__WEBPACK_IMPORTED_MODULE_7__ =
      __webpack_require__(1576)
    /* harmony import */ var _data_siteMetadata__WEBPACK_IMPORTED_MODULE_7___default =
      /*#__PURE__*/ __webpack_require__.n(_data_siteMetadata__WEBPACK_IMPORTED_MODULE_7__)
    /* harmony import */ var _components_comments__WEBPACK_IMPORTED_MODULE_8__ =
      __webpack_require__(896)
    /* harmony import */ var _components_ScrollTopAndComment__WEBPACK_IMPORTED_MODULE_9__ =
      __webpack_require__(7175)
    /* harmony import */ var _fortawesome_react_fontawesome__WEBPACK_IMPORTED_MODULE_10__ =
      __webpack_require__(7197)
    /* harmony import */ var _fortawesome_react_fontawesome__WEBPACK_IMPORTED_MODULE_10___default =
      /*#__PURE__*/ __webpack_require__.n(
        _fortawesome_react_fontawesome__WEBPACK_IMPORTED_MODULE_10__
      )

    const editUrl = (fileName) =>
      `${
        _data_siteMetadata__WEBPACK_IMPORTED_MODULE_7___default().siteRepo
      }/blob/master/data/blog/${fileName}`
    const discussUrl = (slug) =>
      `https://mobile.twitter.com/search?q=${encodeURIComponent(
        `${siteMetadata.siteUrl}/blog/${slug}`
      )}`
    const postDateTemplate = {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    }
    function PostLayout({
      frontMatter,
      authorDetails,
      next,
      prev,
      NotionJsx,
      NotionTitle,
      children,
    }) {
      const { slug, fileName, date, title, images, tags } = frontMatter
      // console.log(children)
      return /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
        _components_SectionContainer__WEBPACK_IMPORTED_MODULE_3__ /* ["default"] */.Z,
        {
          children: [
            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
              _components_SEO__WEBPACK_IMPORTED_MODULE_4__ /* .BlogSEO */.Uy,
              {
                url: `${
                  _data_siteMetadata__WEBPACK_IMPORTED_MODULE_7___default().siteUrl
                }/blog/${slug}`,
                authorDetails: authorDetails,
                ...frontMatter,
              }
            ),
            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
              _components_ScrollTopAndComment__WEBPACK_IMPORTED_MODULE_9__ /* ["default"] */.Z,
              {}
            ),
            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('article', {
              children: /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
                'div',
                {
                  className: 'xl:divide-y xl:divide-gray-200 xl:dark:divide-gray-700',
                  children: [
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('header', {
                      className: 'pt-6 xl:pb-6',
                      children: /*#__PURE__*/ (0,
                      react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('div', {
                        className: 'space-y-1 ',
                        children: [
                          /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('div', {
                            children: /*#__PURE__*/ (0,
                            react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
                              _components_PageTitle__WEBPACK_IMPORTED_MODULE_2__ /* ["default"] */.Z,
                              {
                                children: [
                                  NotionTitle ? NotionTitle : title,
                                  /*#__PURE__*/ (0,
                                  react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('span', {
                                    className: 'text-base',
                                    children: [
                                      /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                        'span',
                                        {
                                          children: ' ',
                                        }
                                      ),
                                      !frontMatter.notion &&
                                        /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                          _components_Link__WEBPACK_IMPORTED_MODULE_1__ /* ["default"] */.Z,
                                          {
                                            href: editUrl(fileName),
                                            className:
                                              'hover:text-primary-600 dark:hover:text-primary-400',
                                            children:
                                              /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                                _fortawesome_react_fontawesome__WEBPACK_IMPORTED_MODULE_10__.FontAwesomeIcon,
                                                {
                                                  icon: 'edit',
                                                  className: 'ml-2',
                                                }
                                              ),
                                          }
                                        ),
                                    ],
                                  }),
                                ],
                              }
                            ),
                          }),
                          /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('dl', {
                            className: 'space-y-10',
                            children:
                              /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                'div',
                                {
                                  children: /*#__PURE__*/ (0,
                                  react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('dd', {
                                    className: 'text-base font-medium leading-6',
                                    children: [
                                      /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                        'span',
                                        {
                                          children: 'Posted by ',
                                        }
                                      ),
                                      /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                        'span',
                                        {
                                          className: 'font-rs italic',
                                          children: authorDetails.map(
                                            (author) => author.name + ' '
                                          ),
                                        }
                                      ),
                                      /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                        'span',
                                        {
                                          children: 'on ',
                                        }
                                      ),
                                      /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                        'time',
                                        {
                                          dateTime: date,
                                          children: new Date(date).toLocaleDateString(
                                            _data_siteMetadata__WEBPACK_IMPORTED_MODULE_7___default()
                                              .locale,
                                            postDateTemplate
                                          ),
                                        }
                                      ),
                                    ],
                                  }),
                                }
                              ),
                          }),
                        ],
                      }),
                    }),
                    /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('div', {
                      className:
                        'divide-y divide-gray-200 pb-8 dark:divide-gray-700 xl:grid xl:grid-cols-4 xl:gap-x-6 xl:divide-y-0',
                      style: {
                        gridTemplateRows: 'auto 1fr',
                      },
                      children: [
                        /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('div', {
                          className:
                            'divide-y divide-gray-200 dark:divide-gray-700 xl:col-span-3 xl:row-span-2 xl:pb-0',
                          children: /*#__PURE__*/ (0,
                          react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('div', {
                            className: 'prose max-w-none pb-8 dark:prose-dark',
                            children: [children, NotionJsx],
                          }),
                        }),
                        /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
                          'footer',
                          {
                            children: [
                              /*#__PURE__*/ (0,
                              react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('div', {
                                className:
                                  'divide-gray-200 text-sm font-medium leading-5 dark:divide-gray-700 xl:col-start-1 xl:row-start-2 xl:divide-y',
                                children: [
                                  tags &&
                                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                      'div',
                                      {
                                        className: 'py-4 xl:py-8',
                                        children:
                                          /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                            'div',
                                            {
                                              className: 'flex flex-wrap',
                                              children: tags.map((tag) =>
                                                /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                                  _components_Tag__WEBPACK_IMPORTED_MODULE_6__ /* ["default"] */.Z,
                                                  {
                                                    text: tag,
                                                  },
                                                  tag
                                                )
                                              ),
                                            }
                                          ),
                                      }
                                    ),
                                  (next || prev) &&
                                    /*#__PURE__*/ (0,
                                    react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('div', {
                                      className:
                                        'flex justify-between py-4 xl:block xl:space-y-8 xl:py-8',
                                      children: [
                                        prev &&
                                          /*#__PURE__*/ (0,
                                          react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
                                            'div',
                                            {
                                              children: [
                                                /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                                  'h2',
                                                  {
                                                    className:
                                                      'text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400',
                                                    children: 'Previous Article',
                                                  }
                                                ),
                                                /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                                  'div',
                                                  {
                                                    className:
                                                      'text-primary-500 hover:text-primary-600 dark:hover:text-primary-400',
                                                    children:
                                                      /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                                        _components_Link__WEBPACK_IMPORTED_MODULE_1__ /* ["default"] */.Z,
                                                        {
                                                          href: `/blog/${prev.slug}`,
                                                          children: prev.title,
                                                        }
                                                      ),
                                                  }
                                                ),
                                              ],
                                            }
                                          ),
                                        next &&
                                          /*#__PURE__*/ (0,
                                          react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
                                            'div',
                                            {
                                              children: [
                                                /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                                  'h2',
                                                  {
                                                    className:
                                                      'text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400',
                                                    children: 'Next Article',
                                                  }
                                                ),
                                                /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                                  'div',
                                                  {
                                                    className:
                                                      'text-primary-500 hover:text-primary-600 dark:hover:text-primary-400',
                                                    children:
                                                      /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                                        _components_Link__WEBPACK_IMPORTED_MODULE_1__ /* ["default"] */.Z,
                                                        {
                                                          href: `/blog/${next.slug}`,
                                                          children: next.title,
                                                        }
                                                      ),
                                                  }
                                                ),
                                              ],
                                            }
                                          ),
                                      ],
                                    }),
                                ],
                              }),
                              /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                'div',
                                {
                                  className: 'pt-4 xl:pt-8',
                                  children:
                                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                      _components_Link__WEBPACK_IMPORTED_MODULE_1__ /* ["default"] */.Z,
                                      {
                                        href: '/blog',
                                        className:
                                          'text-primary-500 hover:text-primary-600 dark:hover:text-primary-400',
                                        children: '\u2190 Back to the blog',
                                      }
                                    ),
                                }
                              ),
                            ],
                          }
                        ),
                      ],
                    }),
                  ],
                }
              ),
            }),
          ],
        }
      )
    }

    /***/
  },

  /***/ 3168: /***/ (__unused_webpack_module, __webpack_exports__, __webpack_require__) => {
    'use strict'
    __webpack_require__.r(__webpack_exports__)
    /* harmony export */ __webpack_require__.d(__webpack_exports__, {
      /* harmony export */ default: () => /* binding */ PostLayout,
      /* harmony export */
    })
    /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ =
      __webpack_require__(997)
    /* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default =
      /*#__PURE__*/ __webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__)
    /* harmony import */ var _components_Link__WEBPACK_IMPORTED_MODULE_1__ =
      __webpack_require__(7233)
    /* harmony import */ var _components_PageTitle__WEBPACK_IMPORTED_MODULE_2__ =
      __webpack_require__(920)
    /* harmony import */ var _components_SectionContainer__WEBPACK_IMPORTED_MODULE_3__ =
      __webpack_require__(890)
    /* harmony import */ var _components_SEO__WEBPACK_IMPORTED_MODULE_4__ =
      __webpack_require__(9831)
    /* harmony import */ var _data_siteMetadata__WEBPACK_IMPORTED_MODULE_5__ =
      __webpack_require__(1576)
    /* harmony import */ var _data_siteMetadata__WEBPACK_IMPORTED_MODULE_5___default =
      /*#__PURE__*/ __webpack_require__.n(_data_siteMetadata__WEBPACK_IMPORTED_MODULE_5__)
    /* harmony import */ var _lib_utils_formatDate__WEBPACK_IMPORTED_MODULE_6__ =
      __webpack_require__(6232)
    /* harmony import */ var _components_comments__WEBPACK_IMPORTED_MODULE_7__ =
      __webpack_require__(896)
    /* harmony import */ var _components_ScrollTopAndComment__WEBPACK_IMPORTED_MODULE_8__ =
      __webpack_require__(7175)

    function PostLayout({ frontMatter, authorDetails, next, prev, children }) {
      const { date, title } = frontMatter
      return /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
        _components_SectionContainer__WEBPACK_IMPORTED_MODULE_3__ /* ["default"] */.Z,
        {
          children: [
            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
              _components_SEO__WEBPACK_IMPORTED_MODULE_4__ /* .BlogSEO */.Uy,
              {
                url: `${_data_siteMetadata__WEBPACK_IMPORTED_MODULE_5___default().siteUrl}/blog/${
                  frontMatter.slug
                }`,
                ...frontMatter,
              }
            ),
            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
              _components_ScrollTopAndComment__WEBPACK_IMPORTED_MODULE_8__ /* ["default"] */.Z,
              {}
            ),
            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('article', {
              children: /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
                'div',
                {
                  children: [
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('header', {
                      children: /*#__PURE__*/ (0,
                      react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('div', {
                        className:
                          'space-y-1 border-b border-gray-200 pb-10 text-center dark:border-gray-700',
                        children: [
                          /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('dl', {
                            children: /*#__PURE__*/ (0,
                            react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('div', {
                              children: [
                                /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                  'dt',
                                  {
                                    className: 'sr-only',
                                    children: 'Published on',
                                  }
                                ),
                                /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                  'dd',
                                  {
                                    className:
                                      'text-base font-medium leading-6 text-gray-500 dark:text-gray-400',
                                    children:
                                      /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                        'time',
                                        {
                                          dateTime: date,
                                          children: (0,
                                          _lib_utils_formatDate__WEBPACK_IMPORTED_MODULE_6__ /* ["default"] */.Z)(
                                            date
                                          ),
                                        }
                                      ),
                                  }
                                ),
                              ],
                            }),
                          }),
                          /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('div', {
                            children:
                              /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                _components_PageTitle__WEBPACK_IMPORTED_MODULE_2__ /* ["default"] */.Z,
                                {
                                  children: title,
                                }
                              ),
                          }),
                        ],
                      }),
                    }),
                    /*#__PURE__*/ (0, react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('div', {
                      className:
                        'divide-y divide-gray-200 pb-8 dark:divide-gray-700 xl:divide-y-0 ',
                      style: {
                        gridTemplateRows: 'auto 1fr',
                      },
                      children: [
                        /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('div', {
                          className:
                            'divide-y divide-gray-200 dark:divide-gray-700 xl:col-span-3 xl:row-span-2 xl:pb-0',
                          children: /*#__PURE__*/ (0,
                          react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('div', {
                            className: 'prose max-w-none pt-10 pb-8 dark:prose-dark',
                            children: ['??', children],
                          }),
                        }),
                        /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                          _components_comments__WEBPACK_IMPORTED_MODULE_7__ /* ["default"] */.Z,
                          {
                            frontMatter: frontMatter,
                          }
                        ),
                        /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx('footer', {
                          children: /*#__PURE__*/ (0,
                          react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)('div', {
                            className:
                              'flex flex-col text-sm font-medium sm:flex-row sm:justify-between sm:text-base',
                            children: [
                              prev &&
                                /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                  'div',
                                  {
                                    className: 'pt-4 xl:pt-8',
                                    children: /*#__PURE__*/ (0,
                                    react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
                                      _components_Link__WEBPACK_IMPORTED_MODULE_1__ /* ["default"] */.Z,
                                      {
                                        href: `/blog/${prev.slug}`,
                                        className:
                                          'text-primary-500 hover:text-primary-600 dark:hover:text-primary-400',
                                        children: ['\u2190 ', prev.title],
                                      }
                                    ),
                                  }
                                ),
                              next &&
                                /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(
                                  'div',
                                  {
                                    className: 'pt-4 xl:pt-8',
                                    children: /*#__PURE__*/ (0,
                                    react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(
                                      _components_Link__WEBPACK_IMPORTED_MODULE_1__ /* ["default"] */.Z,
                                      {
                                        href: `/blog/${next.slug}`,
                                        className:
                                          'text-primary-500 hover:text-primary-600 dark:hover:text-primary-400',
                                        children: [next.title, ' \u2192'],
                                      }
                                    ),
                                  }
                                ),
                            ],
                          }),
                        }),
                      ],
                    }),
                  ],
                }
              ),
            }),
          ],
        }
      )
    }

    /***/
  },
}
