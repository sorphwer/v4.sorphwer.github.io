# Design Guideline | 设计规范

## 1 标准色 | Color Pattern

### 1.1 基础色 | Base

| **#000000** | **#808080**               | **#ffffff** |
| ----------- | ------------------------- | ----------- |
| Pure Black  | 50% Grey (Standard Color) | Pure White  |

### 1.2 装饰色 | Novelty

| **#64d2ff**   | **#0070c9**  |
| ------------- | ------------ |
| RS Blue Light | RS Blue Dark |

### 1.3 警示色 | Aposematism

| **#e83e8c**  |
| ------------ |
| Warning Pink |

### 1.4 配色标准 | Color Ratio Standard

- 基础色、装饰色和警示色的配色比例为 **6 : 3 : 1**。即，产品 60% 的部分应当为黑色、标准灰和白色；30% 的部分应当为蓝色（仅能同时使用一种：亮色调主题下使用 RS Blue Light，反之使用 RS Blue Dark）；10% 的部分应当为警示色。
- Base : Novelty : Aposematism = **6 : 3 : 1**. 60% of the product should be Pure Black, 50% Grey and Pure White; 30% should be blue (only one at a time: RS Blue Light on light themes, RS Blue Dark on dark themes); 10% should be Warning Pink.

### 1.5 CSS 标准色转换代码 | CSS Color Computing from Standard Color

- 在 Web 显示中，可以使用以下 CSS `filter` 将标准灰（#808080）素材转换为其他标准色。因此仅需准备涂色为标准灰的图像素材。请注意颜色转换结果可能因浏览器有所不同。
- On the web, apply the following CSS `filter` values to convert 50% Grey assets into other standard colors, so only grey-colored image assets are required. Results may vary across browsers.

| **Color Name** | **CSS code**                                                                                |
| -------------- | ------------------------------------------------------------------------------------------- |
| Pure Black     | `brightness(0)`                                                                             |
| Pure White     | `brightness(100)`                                                                           |
| Warning Pink   | `invert(51%) sepia(93%) saturate(4588%) hue-rotate(309deg) brightness(94%) contrast(93%)`   |
| RS Blue Dark   | `invert(28%) sepia(53%) saturate(4946%) hue-rotate(192deg) brightness(92%) contrast(101%)`  |
| RS Blue Light  | `invert(70%) sepia(61%) saturate(2249%) hue-rotate(177deg) brightness(112%) contrast(103%)` |

### 1.6 输出对照标准潘通色 | Pantone Mapping for Printing

- 在打印中，请使用以下表格作为参考。
- Use the following mapping as reference for print.

| **Color Name**            | **Pantone Color**   |
| ------------------------- | ------------------- |
| Pure Black                | N/A                 |
| Pure White                | N/A                 |
| 50% Grey (Standard Color) | Pantone 8400 C      |
| Warning Pink              | Pantone 2039 C      |
| RS Blue Dark              | Pantone 285 C       |
| RS Blue Light             | Pantone Blue 0821 C |

## 2 字体 | Fonts

### 2.1 仅用于名称 | For Names Only

| **Scope**                              | **Fonts**                 |
| -------------------------------------- | ------------------------- |
| Nest of Etamine Study, Nest of Etamine | Mistral, Source Han Serif |
| low illumiance, RiinoSite              | ~~Adobe Clean~~ Calibri   |
| RiinoSite Decorator                    | AmarilloUSAF              |

- ~~因为 Adobe Clean 是受限字体，不可公开使用。~~
- Since Adobe Clean is a restricted font, it cannot be used directly in public; use Calibri instead.

### 2.2 仅用于内容 | For Contents Only

| **Standard(s)**                  | **Font Family (CSS)**                                                                                                                                                                                 |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RiinoSite Standard / MS Standard | `font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";` |
| DSP Standard / DTT Standard      | `font-family: Calibri, STXihei;`                                                                                                                                                                      |
