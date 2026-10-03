const TRANSLATIONS: Record<string, string> = {
  '首页': 'Home',
  '持仓': 'Portfolio',
  '机会': 'Opportunities',
  '模拟盘': 'Paper Trading',
  '助手': 'Assistant',
  '提醒': 'Alerts',
  '验证中心': 'Validation',
  '历史': 'History',
  '数据源': 'Data Sources',
  '设置': 'Settings',
  'GitHub 项目': 'GitHub Project',
  '查看日志': 'View Logs',
  '发现新版本': 'Update available',
  '稍后提醒': 'Remind me later',
  '去升级': 'Update',
  'AI 助手': 'AI Assistant',
  '新研究': 'New Research',
  '历史会话': 'Research History',
  '你的研究记录会显示在这里。': 'Your research history will appear here.',
  '今天想研究什么？': 'What do you want to research today?',
  '输入一只股票、一个市场问题，或让 PanWatch 诊断你的持仓。助手会先查询可用数据，再给出有依据的结论。':
    'Enter an asset or market question, or ask PanWatch to diagnose your portfolio. The assistant will research available evidence before answering.',
  '搜索股票，或问：我的持仓风险怎么样？': 'Search an asset or ask: what are the risks in my portfolio?',
  '开始一项研究': 'Start research',
  '发送研究问题': 'Send research question',
  '分析一只股票': 'Analyze an asset',
  '分析一只股票的基本面、行情和近期新闻': 'Analyze an asset using market data, fundamentals and recent news',
  '诊断我的持仓': 'Diagnose my portfolio',
  '诊断我的持仓风险和关键关注点': 'Diagnose my portfolio risks and key watch points',
  '发现今日机会': 'Find today\'s opportunities',
  '结合今天的市场行情，帮我寻找值得研究的机会': 'Use today\'s market conditions to find opportunities worth researching',
  '从标的开始': 'Start with an asset',
  '输入代码或公司名，生成综合、短线或事件驱动分析。': 'Enter a symbol or name for comprehensive, short-term or event-driven analysis.',
  '从持仓开始': 'Start with your portfolio',
  '调用你的实盘和模拟盘数据，识别集中度与风险敞口。': 'Use your live and paper portfolio data to identify concentration and risk exposure.',
  '从问题开始': 'Start with a question',
  '让助手串联行情、K 线和新闻，给出下一步研究方向。': 'Let the assistant connect price action, charts and news into the next research steps.',
  '小助手配置': 'Assistant Settings',
  '管理工具权限，以及上下文压缩使用的模型和预算。': 'Manage tool permissions and context-compression settings.',
  '助手工具权限': 'Assistant tool permissions',
  '读取': 'Read',
  '直接允许': 'Allow',
  '修改': 'Modify',
  '每次询问': 'Ask each time',
  '外部操作': 'External actions',
  '破坏性操作': 'Destructive actions',
  '禁止': 'Deny',
  '查询持仓': 'Read portfolio',
  '查询实时行情': 'Get live quote',
  '发现研究候选': 'Find research candidates',
  '分析 K 线走势': 'Analyze price trend',
  '搜索股票新闻': 'Search asset news',
  '上下文工程': 'Context engineering',
  '配置自动压缩所使用的模型和上下文预算。': 'Configure the model and token budget used for automatic context compression.',
  '上下文压缩模型': 'Context compression model',
  '跟随系统默认模型': 'Use system default model',
  '压缩温度': 'Compression temperature',
  '摘要最大 Token': 'Summary max tokens',
  '最大上下文 Token': 'Max context tokens',
  '自动压缩阈值': 'Auto-compression threshold',
  '硬上限 Token': 'Hard token limit',
  '保留最近消息数': 'Keep recent messages',
  '保存上下文配置': 'Save context settings',
  '保存中…': 'Saving…',
  '已保存': 'Saved',
  '无法加载上下文配置，请稍后重试。': 'Unable to load context settings. Please try again.',
  '保存上下文配置失败，请检查阈值后重试。': 'Failed to save context settings. Check the limits and try again.',
  '设置访问密码': 'Set access password',
  '首次使用，请设置访问密码以保护您的数据': 'For first use, set an access password to protect your data.',
  '用户名': 'Username',
  '请输入用户名': 'Enter username',
  '设置密码': 'Password',
  '至少 6 位': 'At least 6 characters',
  '确认密码': 'Confirm password',
  '再次输入密码': 'Re-enter password',
  '设置密码并进入': 'Set password and enter',
  '欢迎使用盯盘侠': 'Welcome to PanWatch',
  '你的自选股已就绪，可以开始使用了': 'Your watchlist is ready. You can start using PanWatch.',
  '实时行情监控': 'Real-time market monitoring',
  '跟踪自选股价格变动，快速发现异常波动': 'Track watchlist price moves and spot unusual activity quickly.',
  'AI 智能分析': 'AI analysis',
  '盘后日报、异动建议、技术分析': 'Daily reports, unusual-move insights and technical analysis.',
  '智能通知推送': 'Smart notifications',
  '开始使用': 'Get started',
  '跳过引导': 'Skip tour',
  '暂无持仓,添加持仓后这里展示今日盈亏与组合走势': 'No holdings yet. Add positions to see today\'s P&L and portfolio trend.',
  '今日要紧事': 'Key events today',
  '你的持仓/自选里今天该关注的': 'What matters today across your portfolio and watchlist',
  '今日暂无明显异动或触发信号': 'No significant moves or triggered signals today',
  '组合体检': 'Portfolio health',
  '机会精选': 'Top opportunities',
  '进入机会页': 'Open opportunities',
  '暂无活跃机会信号': 'No active opportunity signals',
  '机会发现': 'Opportunity discovery',
  '热门板块': 'Hot sectors',
  '热门股票': 'Hot stocks',
  '涨幅榜': 'Top gainers',
  '系统设置': 'System settings',
  '服务商': 'Providers',
  '模型': 'Models',
  '通知': 'Notifications',
  '运行状态': 'Runtime status',
  '系统自检': 'System check',
  '退出登录': 'Sign out',
  '深色': 'Dark',
  '浅色': 'Light',
  '跟随系统': 'System',
  '搜索': 'Search',
  '刷新': 'Refresh',
  '加载中…': 'Loading…',
  '暂无数据': 'No data',
  '关闭': 'Close',
  '取消': 'Cancel',
  '保存': 'Save',
  '删除': 'Delete',
  '编辑': 'Edit',
  '添加': 'Add',
  '测试': 'Test',
  '成功': 'Success',
  '失败': 'Failed',
}

const ATTRIBUTES = ['placeholder', 'title', 'aria-label'] as const

function translateTextNode(node: Text) {
  const raw = node.textContent || ''
  const match = raw.match(/^(\s*)(.*?)(\s*)$/s)
  if (!match) return
  const translated = TRANSLATIONS[match[2]]
  if (translated && translated !== match[2]) {
    node.textContent = `${match[1]}${translated}${match[3]}`
  }
}

function translateElement(element: Element) {
  for (const attr of ATTRIBUTES) {
    const value = element.getAttribute(attr)
    if (value && TRANSLATIONS[value]) element.setAttribute(attr, TRANSLATIONS[value])
  }
  for (const child of Array.from(element.childNodes)) {
    if (child.nodeType === Node.TEXT_NODE) {
      translateTextNode(child as Text)
    } else if (child.nodeType === Node.ELEMENT_NODE) {
      translateElement(child as Element)
    }
  }
}

export function installEnglishUI() {
  document.documentElement.lang = 'en'

  const translateDocument = () => {
    if (document.body) translateElement(document.body)
  }

  translateDocument()

  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type === 'characterData' && mutation.target.nodeType === Node.TEXT_NODE) {
        translateTextNode(mutation.target as Text)
        continue
      }
      for (const node of Array.from(mutation.addedNodes)) {
        if (node.nodeType === Node.TEXT_NODE) translateTextNode(node as Text)
        if (node.nodeType === Node.ELEMENT_NODE) translateElement(node as Element)
      }
      if (mutation.type === 'attributes' && mutation.target.nodeType === Node.ELEMENT_NODE) {
        translateElement(mutation.target as Element)
      }
    }
  })

  observer.observe(document.documentElement, {
    subtree: true,
    childList: true,
    characterData: true,
    attributes: true,
    attributeFilter: [...ATTRIBUTES],
  })
}
