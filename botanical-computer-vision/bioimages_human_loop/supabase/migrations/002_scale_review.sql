-- Shared anonymous review for the 92-image photographic-scale audit.
-- Anonymous Supabase users still receive auth.uid() and the authenticated role.

create table if not exists public.scale_review_batches (
  id uuid primary key,
  slug text unique not null,
  title text not null,
  status text not null default 'open' check (status in ('open', 'closed')),
  created_at timestamptz not null default now()
);

create table if not exists public.scale_review_items (
  batch_id uuid not null references public.scale_review_batches(id) on delete cascade,
  image_id text not null,
  display_order integer not null,
  species text not null,
  organ_category text not null,
  subview text not null,
  provisional_scale text not null,
  provisional_confidence text not null,
  primary key (batch_id, image_id),
  unique (batch_id, display_order)
);

create table if not exists public.scale_annotations (
  batch_id uuid not null,
  image_id text not null,
  reviewer_id uuid not null default auth.uid() references auth.users(id),
  human_scale text not null check (human_scale in ('distant', 'mid-range', 'close-up', 'uncertain')),
  confidence text not null check (confidence in ('high', 'medium', 'low')),
  note text,
  revision integer not null default 1,
  submitted_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (batch_id, image_id, reviewer_id),
  foreign key (batch_id, image_id)
    references public.scale_review_items(batch_id, image_id) on delete cascade
);

create table if not exists public.scale_annotation_events (
  id bigint generated always as identity primary key,
  batch_id uuid not null,
  image_id text not null,
  reviewer_id uuid not null references auth.users(id),
  payload jsonb not null,
  created_at timestamptz not null default now()
);

alter table public.scale_review_batches enable row level security;
alter table public.scale_review_items enable row level security;
alter table public.scale_annotations enable row level security;
alter table public.scale_annotation_events enable row level security;

drop policy if exists "scale batches readable" on public.scale_review_batches;
create policy "scale batches readable" on public.scale_review_batches
  for select to authenticated using (true);

drop policy if exists "scale items readable" on public.scale_review_items;
create policy "scale items readable" on public.scale_review_items
  for select to authenticated using (true);

drop policy if exists "reviewers read own scale annotations" on public.scale_annotations;
create policy "reviewers read own scale annotations" on public.scale_annotations
  for select to authenticated using (reviewer_id = auth.uid());

drop policy if exists "reviewers insert own scale annotations" on public.scale_annotations;
create policy "reviewers insert own scale annotations" on public.scale_annotations
  for insert to authenticated with check (reviewer_id = auth.uid());

drop policy if exists "reviewers update own scale annotations" on public.scale_annotations;
create policy "reviewers update own scale annotations" on public.scale_annotations
  for update to authenticated using (reviewer_id = auth.uid()) with check (reviewer_id = auth.uid());

drop policy if exists "reviewers read own scale events" on public.scale_annotation_events;
create policy "reviewers read own scale events" on public.scale_annotation_events
  for select to authenticated using (reviewer_id = auth.uid());

drop policy if exists "reviewers insert own scale events" on public.scale_annotation_events;
create policy "reviewers insert own scale events" on public.scale_annotation_events
  for insert to authenticated with check (reviewer_id = auth.uid());

grant select on public.scale_review_batches, public.scale_review_items to authenticated;
grant select, insert, update on public.scale_annotations to authenticated;
grant select, insert on public.scale_annotation_events to authenticated;
revoke delete on public.scale_annotations, public.scale_annotation_events from anon, authenticated;

create or replace function public.submit_scale_annotation(
  p_batch_id uuid,
  p_image_id text,
  p_human_scale text,
  p_confidence text,
  p_note text default null
) returns public.scale_annotations
language plpgsql
security invoker
set search_path = public
as $$
declare
  result public.scale_annotations;
begin
  if auth.uid() is null then
    raise exception 'Authentication required';
  end if;

  insert into public.scale_annotations (
    batch_id, image_id, reviewer_id, human_scale, confidence, note
  ) values (
    p_batch_id, p_image_id, auth.uid(), p_human_scale, p_confidence, nullif(trim(p_note), '')
  )
  on conflict (batch_id, image_id, reviewer_id) do update set
    human_scale = excluded.human_scale,
    confidence = excluded.confidence,
    note = excluded.note,
    revision = public.scale_annotations.revision + 1,
    updated_at = now()
  returning * into result;

  insert into public.scale_annotation_events (batch_id, image_id, reviewer_id, payload)
  values (p_batch_id, p_image_id, auth.uid(), to_jsonb(result));

  return result;
end;
$$;

revoke all on function public.submit_scale_annotation(uuid, text, text, text, text) from public, anon;
grant execute on function public.submit_scale_annotation(uuid, text, text, text, text) to authenticated;

insert into public.scale_review_batches (id, slug, title)
values ('00000000-0000-4000-8000-000000000002', 'scale-audit-v1', 'BioImages photographic-scale audit')
on conflict (id) do update set title = excluded.title;

insert into public.scale_review_items (
  batch_id, image_id, display_order, species, organ_category, subview,
  provisional_scale, provisional_confidence
) values
('00000000-0000-4000-8000-000000000002', 'baskauf/11614', 1, 'Cornus florida', 'leaf', 'whole upper surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11695', 2, 'Lindera benzoin', 'leaf', 'whole upper surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/20454', 3, 'Toxicodendron radicans', 'leaf', 'whole upper surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/39171', 4, 'Ailanthus altissima', 'twig', 'close-up winter terminal bud', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/39176', 5, 'Fagus grandifolia', 'twig', 'close-up winter terminal bud', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/49248', 6, 'Betula alleghaniensis', 'leaf', 'margin of upper + lower surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/52662', 7, 'Rhus aromatica', 'leaf', 'margin of upper + lower surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/b5068', 8, 'Carpinus caroliniana', 'leaf', 'whole upper surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2014', 9, 'Quercus palustris', 'leaf', 'margin of upper + lower surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2356', 10, 'Quercus palustris', 'leaf', 'margin of upper + lower surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11174', 11, 'Ginkgo biloba', 'leaf', 'showing orientation on twig', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11902', 12, 'Gymnocladus dioicus', 'bark', 'of a small tree or small branch', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/12438', 13, 'Quercus palustris', 'bark', 'of a large tree', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/12467', 14, 'Fraxinus americana', 'leaf', 'unspecified', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/17455', 15, 'Fraxinus pennsylvanica', 'twig', 'winter overall', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/26143', 16, 'Sambucus racemosa', 'leaf', 'whole upper surface', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/29699', 17, 'Juglans nigra', 'twig', 'unspecified', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/ac1449', 18, 'Quercus alba', 'bark', 'of a large tree', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2095', 19, 'Quercus alba', 'twig', 'orientation of petioles', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'thomas/0049-08-02', 20, 'Cercis canadensis', 'inflorescence', 'whole - unspecified', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10643', 21, 'Prunus serotina', 'whole tree (or vine)', 'view up trunk', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11497', 22, 'Tsuga canadensis', 'whole tree', 'view up trunk', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11930', 23, 'Acer rubrum', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/12321', 24, 'Quercus muehlenbergii', 'whole tree (or vine)', 'view up trunk', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/35191', 25, 'Quercus alba', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/90661', 26, 'Quercus macrocarpa', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/90696', 27, 'Quercus rubra', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/90826a', 28, 'Fraxinus americana', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/91091', 29, 'Fraxinus pennsylvanica', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'ssmv/2-638-02', 30, 'Fraxinus americana', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10897', 31, 'Acer negundo', 'whole tree (or vine)', 'general', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10904', 32, 'Ailanthus altissima', 'whole tree (or vine)', 'general', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11086', 33, 'Lindera benzoin', 'whole tree (or vine)', 'general', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/13516', 34, 'Rhus glabra', 'fruit', 'as borne on the plant', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/15728', 35, 'Carpinus caroliniana', 'bark', 'of a small tree or small branch', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/19579', 36, 'Ulmus rubra', 'fruit', 'as borne on the plant', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/30778', 37, 'Viburnum opulus', 'whole tree (or vine)', 'winter', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/35713', 38, 'Zanthoxylum americanum', 'whole tree (or vine)', 'general', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/43553', 39, 'Thuja occidentalis', 'whole tree', 'view up trunk', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/50749', 40, 'Acer rubrum', 'whole tree (or vine)', 'general', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10414', 41, 'Rhus glabra', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/12700', 42, 'Rhus aromatica', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/15935', 43, 'Juglans nigra', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/16835', 44, 'Platanus occidentalis', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/2014-08-23-16-40-31', 45, 'Fraxinus quadrangulata', 'leaf', 'showing orientation on twig', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/28499', 46, 'Acer negundo', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/50392', 47, 'Picea glauca', 'bark', 'of a medium tree or large branch', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/b5071', 48, 'Carpinus caroliniana', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em1998', 49, 'Fagus grandifolia', 'twig', 'orientation of petioles', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2316', 50, 'Quercus palustris', 'leaf', 'showing orientation on twig', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10503', 51, 'Fagus grandifolia', 'inflorescence', 'whole - male', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10675', 52, 'Aesculus hippocastanum', 'inflorescence', 'whole - unspecified', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/16046', 53, 'Acer negundo', 'twig', 'unspecified', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/19171', 54, 'Carpinus caroliniana', 'inflorescence', 'whole - unspecified', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/31283', 55, 'Pyrus calleryana', 'twig', 'winter overall', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/51172', 56, 'Fraxinus pennsylvanica', 'inflorescence', 'whole - female', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/51572', 57, 'Carya cordiformis', 'inflorescence', 'whole - unspecified', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/66138', 58, 'Gleditsia triacanthos', 'inflorescence', 'whole - female', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/66183', 59, 'Gleditsia triacanthos', 'inflorescence', 'whole - male', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/66184', 60, 'Gleditsia triacanthos', 'inflorescence', 'whole - male', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11464', 61, 'Tilia americana', 'leaf', 'showing orientation on twig', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11899', 62, 'Gymnocladus dioicus', 'leaf', 'unspecified', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11945', 63, 'Aesculus hippocastanum', 'whole tree (or vine)', 'general', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/16712', 64, 'Nyssa sylvatica', 'whole tree (or vine)', 'view up trunk', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/16720', 65, 'Fagus grandifolia', 'leaf', 'showing orientation on twig', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/24248', 66, 'Gymnocladus dioicus', 'leaf', 'whole upper surface', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/26495', 67, 'Viburnum lantanoides', 'whole tree (or vine)', 'general', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/31257', 68, 'Pyrus calleryana', 'whole tree (or vine)', 'winter', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/32045', 69, 'Pyrus calleryana', 'whole tree (or vine)', 'general', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/32863', 70, 'Fraxinus quadrangulata', 'whole tree (or vine)', 'general', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'kaufmannm/ke062', 71, 'Populus tremuloides', 'whole tree (or vine)', 'general', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10462', 72, 'Platanus occidentalis', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10499', 73, 'Quercus rubra', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11251', 74, 'Quercus alba', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11449', 75, 'Betula alleghaniensis', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11458', 76, 'Robinia pseudoacacia', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/13650', 77, 'Carya ovata', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/13843', 78, 'Carya laciniosa', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/16502', 79, 'Juglans cinerea', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/16842', 80, 'Quercus alba', 'seed', 'unspecified', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/20950', 81, 'Ulmus americana', 'fruit', 'unspecified', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/26421', 82, 'Betula alleghaniensis', 'inflorescence', 'whole - male', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/27896', 83, 'Carya cordiformis', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/27897', 84, 'Carya cordiformis', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/50500', 85, 'Pinus resinosa', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/79653', 86, 'Quercus macrocarpa', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/89955', 87, 'Platanus occidentalis', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'bassettst/sb569', 88, 'Betula papyrifera', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/ac1448', 89, 'Quercus alba', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/ac1469', 90, 'Quercus bicolor', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2106', 91, 'Quercus alba', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2192', 92, 'Fagus grandifolia', 'bark', 'of a large tree', 'uncertain', 'low')
on conflict (batch_id, image_id) do update set
  display_order = excluded.display_order,
  species = excluded.species,
  organ_category = excluded.organ_category,
  subview = excluded.subview,
  provisional_scale = excluded.provisional_scale,
  provisional_confidence = excluded.provisional_confidence;
