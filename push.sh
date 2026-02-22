#!/bin/bash

# 自动push脚本 - 一键提交并推送本地更改到远端
# commit信息格式：日期时间 + 今天的第几次提交
# 使用方法: ./push.sh [分支名] [远程仓库名]
# 示例: ./push.sh main origin

# 设置错误时退出
set -e

# 获取分支名（如果通过参数指定，否则使用当前分支）
if [ -n "$1" ]; then
    BRANCH="$1"
    echo "使用指定的分支: ${BRANCH}"
    # 检查分支是否存在
    if ! git show-ref --verify --quiet refs/heads/"${BRANCH}"; then
        echo "❌ 错误: 分支 '${BRANCH}' 不存在"
        exit 1
    fi
    # 切换到指定分支
    git checkout "${BRANCH}"
else
    # 自动检测当前分支
    BRANCH=$(git branch --show-current)
    if [ -z "${BRANCH}" ]; then
        echo "❌ 错误: 无法检测当前分支"
        exit 1
    fi
fi

# 获取远程仓库名（默认为origin）
REMOTE="${2:-origin}"

# 检查远程仓库是否存在
if ! git remote get-url "${REMOTE}" > /dev/null 2>&1; then
    echo "❌ 错误: 远程仓库 '${REMOTE}' 不存在"
    exit 1
fi

echo "当前分支: ${BRANCH}"
echo "远程仓库: ${REMOTE}"
echo ""

# 获取当前日期时间（中文格式）
CURRENT_DATE=$(date +"%Y年%m月%d日")
CURRENT_TIME=$(date +"%H:%M:%S")
DATETIME="${CURRENT_DATE} ${CURRENT_TIME}"

# 获取今天的日期（用于统计提交次数）
TODAY=$(date +"%Y-%m-%d")

# 统计今天已经有多少次提交
COMMIT_COUNT=$(git log --since="${TODAY} 00:00:00" --until="${TODAY} 23:59:59" --oneline | wc -l | tr -d ' ')

# 计算本次是今天的第几次提交
COMMIT_NUMBER=$((COMMIT_COUNT + 1))

# 生成commit信息
COMMIT_MSG="${DATETIME} - 今日第${COMMIT_NUMBER}次提交"

# 检查是否有未提交的更改
if [ -z "$(git status --porcelain)" ]; then
    echo "没有需要提交的更改"
    exit 0
fi

# 显示将要提交的文件
echo "========================================="
echo "准备提交以下更改："
echo "========================================="
git status -s
echo ""

# 添加所有更改
echo "正在添加所有更改..."
git add .

# 提交更改
echo "正在提交更改..."
echo "Commit信息: ${COMMIT_MSG}"
git commit -m "${COMMIT_MSG}"

# 推送到远端
echo "正在推送到远端 ${REMOTE}/${BRANCH}..."
git push "${REMOTE}" "${BRANCH}"

# 如果当前分支没有设置上游分支，设置上游分支
if [ -z "$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null)" ]; then
    echo "设置上游分支: ${REMOTE}/${BRANCH}"
    git branch --set-upstream-to="${REMOTE}/${BRANCH}" "${BRANCH}"
fi

echo ""
echo "========================================="
echo "✅ 提交成功！"
echo "分支: ${BRANCH}"
echo "远程: ${REMOTE}"
echo "Commit信息: ${COMMIT_MSG}"
echo "========================================="
