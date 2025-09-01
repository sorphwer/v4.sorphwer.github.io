# 本地构建后部署到 Vercel 的方法

## 方案 1：使用 Vercel CLI 直接部署（推荐）

### 1. 安装 Vercel CLI

```bash
npm install -g vercel
```

### 2. 登录 Vercel

```bash
vercel login
```

### 3. 本地构建项目

```bash
npm run build
```

### 4. 部署到 Vercel（跳过构建步骤）

```bash
vercel --prebuilt
```

`--prebuilt` 参数告诉 Vercel 直接使用本地的 `.next` 文件夹，不进行远程构建。

## 方案 2：修改 package.json 脚本（适用于自动化部署）

在 `package.json` 中添加一个新的构建脚本：

```json
{
  "scripts": {
    "build:vercel": "echo 'Using pre-built files' && exit 0"
  }
}
```

然后在 Vercel 项目设置中：

1. 进入项目的 Settings
2. 找到 Build & Development Settings
3. 将 Build Command 改为 `npm run build:vercel`
4. Output Directory 设置为 `.next`

## 方案 3：创建 vercel.json 配置文件

创建 `vercel.json` 文件：

```json
{
  "buildCommand": "echo 'Skip build'",
  "outputDirectory": ".next",
  "framework": "nextjs"
}
```

## 注意事项

1. 确保 `.next` 文件夹已经被添加到 git 中（通常它在.gitignore 中）
2. 如果使用方案 1，每次更新代码后需要重新运行 `npm run build` 然后 `vercel --prebuilt`
3. 推荐使用方案 1，因为它最直接且可控
