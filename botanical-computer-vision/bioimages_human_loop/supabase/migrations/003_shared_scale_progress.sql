-- Shared progress for scale review only. Keep per-reviewer writes and audit history.
-- Apply after 002_scale_review.sql; existing annotations remain intact.
begin;
drop policy if exists "reviewers read own scale annotations" on public.scale_annotations;
drop policy if exists "reviewers read shared scale annotations" on public.scale_annotations;
create policy "reviewers read shared scale annotations" on public.scale_annotations
  for select to authenticated using (true);
commit;
