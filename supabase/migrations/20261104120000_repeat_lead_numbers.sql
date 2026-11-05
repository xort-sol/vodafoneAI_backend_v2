-- shortcut: one number may be several leads, so a test campaign can ring one phone many times; the dialler still never
-- rings a number twice at once. Restore a per-campaign unique (campaign_id, phone) before importing real customers.
alter table public.leads drop constraint leads_phone_key;
create index leads_phone on public.leads (phone);
