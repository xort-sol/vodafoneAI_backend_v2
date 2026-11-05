// Paste into your Google Sheet: Extensions > Apps Script. Then Deploy > New deployment > Web app,
// "Execute as: Me", "Who has access: Anyone". Copy the web app URL into RESULTS_WEBHOOK_URL.
// TOKEN must match RESULTS_WEBHOOK_TOKEN in the agent's .env, so only the agent can add rows.
// Don't press "Run" in the editor: doPost only gets data from a real web request, so Run fails with "reading 'postData'".
const TOKEN = 'PASTE_RESULTS_WEBHOOK_TOKEN_HERE';
const COLUMNS = ['timestamp', 'name', 'phone', 'company', 'outcome', 'interested_services', 'callback_time', 'notes', 'room'];

function doPost(e) {
  const data = JSON.parse(e.postData.contents);
  if (data.token !== TOKEN) return ContentService.createTextOutput('forbidden');

  const lock = LockService.getScriptLock();  // two calls ending at once must not both write the header
  lock.waitLock(10000);
  try {
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
    if (sheet.getLastRow() === 0) sheet.appendRow(COLUMNS);
    // Text from the conversation starting with = + - @ would run as a formula; the leading ' keeps it plain text
    sheet.appendRow(COLUMNS.map(c => String(data[c] ?? '').replace(/^([=+\-@])/, "'$1")));
  } finally {
    lock.releaseLock();
  }
  return ContentService.createTextOutput('ok');
}
