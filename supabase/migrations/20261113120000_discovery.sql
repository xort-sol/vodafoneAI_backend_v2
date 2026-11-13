-- Discovery-first calls: the facts a call captured, the matrix classification, three new outcomes and their follow-up details.

alter table public.analyses
  add column discovery jsonb,
  add column classification text check (classification in ('hot', 'warm', 'developing', 'low'));

-- auto-named by the inline check in 20261030120000_phase2_schema.sql
alter table public.calls drop constraint calls_outcome_check;
alter table public.calls
  add constraint calls_outcome_check check (outcome in ('interested', 'callback', 'not_interested', 'do_not_call', 'hung_up',
                                                        'no_answer', 'voicemail', 'unreachable', 'no_response',
                                                        'transferred', 'field_visit', 'email_request')),
  add column follow_up jsonb;
