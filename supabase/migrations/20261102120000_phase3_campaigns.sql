-- Phase 3: leads belong to a campaign and carry the dialler's state; managers write campaigns and leads from the web app.

alter table public.leads
  add column campaign_id bigint references public.campaigns,
  add column attempts int not null default 0,
  add column next_attempt_at timestamptz,
  add column dialled_at timestamptz,
  add constraint leads_status_check check (status in
    ('new', 'dialling', 'retry', 'callback', 'interested', 'closed', 'exhausted', 'do_not_call'));

create index leads_campaign_status on public.leads (campaign_id, status);

-- When the customer asked to be called back, as a timestamp the dialler can act on (callback_time stays the spoken words)
alter table public.calls add column callback_at timestamptz;

create policy "managers create" on public.campaigns for insert
  with check (public.app_role() in ('admin', 'sales_manager'));
create policy "managers change" on public.campaigns for update
  using (public.app_role() in ('admin', 'sales_manager')) with check (public.app_role() in ('admin', 'sales_manager'));
create policy "managers import" on public.leads for insert
  with check (public.app_role() in ('admin', 'sales_manager'));
create policy "team logs own actions" on public.audit_log for insert
  with check (public.app_role() is not null and actor = auth.uid());
