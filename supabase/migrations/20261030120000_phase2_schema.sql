-- Phase 2 schema. Writes come from the agent's server-side connection (table owner, bypasses RLS);
-- signed-in web app users only read, under the policies at the bottom.

create table public.profiles (
  id uuid primary key references auth.users on delete cascade,
  email text,
  full_name text,
  -- No role until an admin gives one: Supabase allows sign-ups by default, and every policy needs a role
  role text check (role in ('admin', 'sales_manager', 'salesperson', 'analyst')),
  created_at timestamptz not null default now()
);

create table public.leads (
  id bigint generated always as identity primary key,
  name text,
  phone text unique,
  company text,
  staff int,
  current_plan text,
  contract_end text,
  notes text,
  status text not null default 'new',
  created_at timestamptz not null default now()
);

create table public.products (
  id bigint generated always as identity primary key,
  name text not null,
  category text,
  price_text text,
  details text,
  active boolean not null default true,
  updated_at timestamptz not null default now()
);

create table public.campaigns (
  id bigint generated always as identity primary key,
  name text not null,
  status text not null default 'draft' check (status in ('draft', 'approved', 'running', 'paused', 'done')),
  calling_hours_start time,
  calling_hours_end time,
  daily_cap int,
  created_by uuid references public.profiles,
  approved_by uuid references public.profiles,
  created_at timestamptz not null default now()
);

create table public.calls (
  id bigint generated always as identity primary key,
  job_id text not null unique,  -- LiveKit job id; room names repeat (every console call is "console-room")
  room text,
  phone text,
  name text,
  company text,
  lead_id bigint references public.leads,
  campaign_id bigint references public.campaigns,
  outcome text not null check (outcome in ('interested', 'callback', 'not_interested', 'do_not_call', 'hung_up',
                                           'no_answer', 'voicemail', 'unreachable', 'no_response')),
  interested_services text,
  callback_time text,
  notes text,
  created_at timestamptz not null default now()
);

create table public.transcripts (
  call_id bigint not null references public.calls on delete cascade,
  position int not null,
  role text not null check (role in ('agent', 'customer')),
  text text not null,
  primary key (call_id, position)
);

create table public.analyses (
  call_id bigint primary key references public.calls on delete cascade,
  score int check (score between 0 and 100),
  reason text,
  objections text[],
  sentiment text,
  next_step text,
  brief text,
  model text,
  created_at timestamptz not null default now()
);

create table public.tasks (
  id bigint generated always as identity primary key,
  lead_id bigint references public.leads,
  call_id bigint references public.calls,
  assigned_to uuid references public.profiles,
  kind text not null check (kind in ('callback', 'follow_up')),
  due_at timestamptz,
  done boolean not null default false,
  created_at timestamptz not null default now()
);

create table public.suppression (
  phone text primary key,
  reason text,
  source text not null check (source in ('call', 'manual')),
  created_at timestamptz not null default now()
);

create table public.audit_log (
  id bigint generated always as identity primary key,
  actor uuid,
  action text not null,
  entity text,
  entity_id text,
  details jsonb,
  at timestamptz not null default now()
);

-- The caller's role, or null when signed out or without a profile. Security definer so policies on
-- profiles can call it without recursing into profiles' own policies.
create function public.app_role() returns text
language sql stable security definer set search_path = '' as $$
  select role from public.profiles where id = auth.uid()
$$;

create function public.handle_new_user() returns trigger
language plpgsql security definer set search_path = '' as $$
begin
  insert into public.profiles (id, email) values (new.id, new.email);
  return new;
end
$$;

create trigger on_auth_user_created after insert on auth.users
  for each row execute function public.handle_new_user();

alter table public.profiles enable row level security;
alter table public.leads enable row level security;
alter table public.products enable row level security;
alter table public.campaigns enable row level security;
alter table public.calls enable row level security;
alter table public.transcripts enable row level security;
alter table public.analyses enable row level security;
alter table public.tasks enable row level security;
alter table public.suppression enable row level security;
alter table public.audit_log enable row level security;

create policy "own profile, or all for managers" on public.profiles for select
  using (id = auth.uid() or public.app_role() in ('admin', 'sales_manager'));
create policy "admins change roles" on public.profiles for update
  using (public.app_role() = 'admin') with check (public.app_role() = 'admin');

create policy "signed-in team reads" on public.leads for select using (public.app_role() is not null);
create policy "signed-in team reads" on public.products for select using (public.app_role() is not null);
create policy "signed-in team reads" on public.campaigns for select using (public.app_role() is not null);
create policy "signed-in team reads" on public.calls for select using (public.app_role() is not null);
create policy "signed-in team reads" on public.transcripts for select using (public.app_role() is not null);
create policy "signed-in team reads" on public.analyses for select using (public.app_role() is not null);
create policy "signed-in team reads" on public.tasks for select using (public.app_role() is not null);
create policy "managers read" on public.suppression for select using (public.app_role() in ('admin', 'sales_manager'));
create policy "admins read" on public.audit_log for select using (public.app_role() = 'admin');
