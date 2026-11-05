-- Phase 4: call cost and latency, a claimable lead inbox (tasks) and the dashboard's daily numbers.

alter table public.calls
  add column duration_s int,
  add column avg_latency_ms int;

alter table public.tasks add column claimed_at timestamptz;
-- one follow-up per call, so re-running the analysis can't put a lead in the inbox twice
create unique index tasks_one_follow_up_per_call on public.tasks (call_id) where kind = 'follow_up';

-- Shared inbox: a salesperson claims an unclaimed task or works their own; managers and admins reassign
create policy "claim or work own task" on public.tasks for update
  using (public.app_role() in ('admin', 'sales_manager')
         or (public.app_role() = 'salesperson' and (assigned_to is null or assigned_to = auth.uid())))
  with check (public.app_role() in ('admin', 'sales_manager')
              or (public.app_role() = 'salesperson' and assigned_to = auth.uid()));

-- security_invoker: the view reads calls under the viewer's RLS, like the table itself
create view public.dashboard_daily with (security_invoker = true) as
select
  (created_at at time zone 'Europe/Dublin')::date as day,
  count(*) as dialled,
  count(*) filter (where outcome not in ('no_answer', 'voicemail', 'unreachable')) as reached,
  count(*) filter (where outcome = 'interested') as interested,
  count(*) filter (where outcome = 'do_not_call') as opt_outs,
  round(coalesce(sum(duration_s), 0) / 60.0, 1) as minutes,
  round(avg(avg_latency_ms)) as avg_latency_ms
from public.calls
group by 1;
