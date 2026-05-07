// background.js — service worker
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'CAPTURE_TAB') {
    chrome.tabs.captureVisibleTab(
      sender.tab?.windowId ?? chrome.windows.WINDOW_ID_CURRENT,
      { format: 'jpeg', quality: 95 },
      dataUrl => {
        if (chrome.runtime.lastError) sendResponse(null)
        else sendResponse(dataUrl)
      }
    )
    return true
  }
  if (msg.type === 'OPEN_POPUP') {
    chrome.action.openPopup?.().catch(() => {})
    sendResponse(true)
  }
  if (msg.type === 'VERIFY_STEP') {
    // Forward step to popup if open
    chrome.runtime.sendMessage(msg).catch(() => {})
    sendResponse(true)
  }
})
