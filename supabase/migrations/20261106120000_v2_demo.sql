-- Schema for the v2 frontend: lead and campaign fields, notes, and column-level lead writes from the web.

alter table public.leads
  add column email text,
  add column city text,
  add column customer_type text,
  add column source text,
  add column tags text[] not null default '{}',
  add column consent boolean not null default false,
  add column archived boolean not null default false,
  add column assigned_to uuid references public.profiles,
  add column sales_status text,
  add column score int,
  add column sentiment text,
  add column ai_summary text,
  add column last_call_at timestamptz,
  add column next_follow_up timestamptz,
  add column product_ref text;

-- A window that ends before it starts can never be open; fix old rows so the check can be added
update public.campaigns set calling_hours_end = calling_hours_start + interval '1 hour'
  where calling_hours_end <= calling_hours_start;

alter table public.campaigns
  add column product_ref text,
  add column agent_ref text,
  add column offer_ref text,
  add column description text,
  add column segment text,
  add column start_date timestamptz,
  add column end_date timestamptz,
  add column days text[] not null default '{Mon,Tue,Wed,Thu,Fri}',
  add column max_attempts int not null default 3,
  add column retry_hours int not null default 4,
  add column concurrency int not null default 2,
  add column min_score int not null default 70,
  add constraint campaigns_hours_order check (calling_hours_end > calling_hours_start);

alter table public.transcripts add column t_seconds int;

alter table public.analyses
  add column questions text[],
  add column unanswered text[],
  add column qualification jsonb;

create table public.lead_notes (
  id bigint generated always as identity primary key,
  lead_id bigint not null references public.leads on delete cascade,
  author text not null,
  text text not null,
  created_at timestamptz not null default now()
);

alter table public.lead_notes enable row level security;
create policy "signed-in team reads" on public.lead_notes for select using (public.app_role() is not null);
create policy "sellers add notes" on public.lead_notes for insert
  with check (public.app_role() in ('admin', 'sales_manager', 'salesperson'));

-- The web edits only these lead columns, never the dialler's status, attempts, next_attempt_at or dialled_at
create policy "sellers change leads" on public.leads for update
  using (public.app_role() in ('admin', 'sales_manager', 'salesperson'))
  with check (public.app_role() in ('admin', 'sales_manager', 'salesperson'));
revoke update on public.leads from anon, authenticated;
grant update (name, email, city, customer_type, tags, consent, archived, assigned_to, sales_status,
              next_follow_up, product_ref, campaign_id) on public.leads to authenticated;

-- The team picks assignees by name, so every role reads every profile
drop policy "own profile, or all for managers" on public.profiles;
create policy "team reads profiles" on public.profiles for select
  using (public.app_role() is not null or id = auth.uid());

create policy "sellers add manual suppression" on public.suppression for insert
  with check (public.app_role() in ('admin', 'sales_manager', 'salesperson') and source = 'manual');
create policy "managers remove suppression" on public.suppression for delete
  using (public.app_role() in ('admin', 'sales_manager'));
