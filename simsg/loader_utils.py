from simsg.data.vg import SceneGraphNoPairsDataset, collate_fn_nopairs
from simsg.data.coco import CocoSceneGraphDataset, coco_collate_fn
from simsg.data.clevr import SceneGraphWithPairsDataset, collate_fn_withpairs

import json
from torch.utils.data import DataLoader


def build_clevr_supervised_train_dsets(args):
  print("building fully supervised %s dataset" % args.dataset)
  with open(args.vocab_json, 'r') as f:
    vocab = json.load(f)
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.train_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.image_size,
    'max_samples': args.num_train_samples,
    'max_objects': args.max_objects_per_image,
    'use_orphaned_objects': args.vg_use_orphaned_objects,
    'include_relationships': args.include_relationships,
  }
  train_dset = SceneGraphWithPairsDataset(**dset_kwargs)
  iter_per_epoch = len(train_dset) // args.batch_size
  print('There are %d iterations per epoch' % iter_per_epoch)

  dset_kwargs['h5_path'] = args.val_h5
  del dset_kwargs['max_samples']
  val_dset = SceneGraphWithPairsDataset(**dset_kwargs)

  dset_kwargs['h5_path'] = args.test_h5
  test_dset = SceneGraphWithPairsDataset(**dset_kwargs)

  return vocab, train_dset, val_dset, test_dset


def build_dset_nopairs(args, checkpoint):

  vocab = checkpoint['model_kwargs']['vocab']
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.data_h5,
    'image_dir': args.data_image_dir,
    'image_size': args.image_size,
    'max_objects': checkpoint['args']['max_objects_per_image'],
    'use_orphaned_objects': checkpoint['args']['vg_use_orphaned_objects'],
    'mode': args.mode,
    'predgraphs': args.predgraphs
  }
  dset = SceneGraphNoPairsDataset(**dset_kwargs)

  return dset


def build_dset_withpairs(args, checkpoint, vocab_t):

  vocab = vocab_t
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.data_h5,
    'image_dir': args.data_image_dir,
    'image_size': args.image_size,
    'max_objects': checkpoint['args']['max_objects_per_image'],
    'use_orphaned_objects': checkpoint['args']['vg_use_orphaned_objects'],
    'mode': args.mode
  }
  dset = SceneGraphWithPairsDataset(**dset_kwargs)

  return dset


def build_eval_loader(args, checkpoint, vocab_t=None, no_gt=False):

  if args.dataset == 'vg' or (no_gt and args.dataset == 'clevr'):
    dset = build_dset_nopairs(args, checkpoint)
    collate_fn = collate_fn_nopairs
  if args.dataset == 'coco':
    _, _, dset = build_coco_dsets(args)
    collate_fn = coco_collate_fn
  elif args.dataset == 'clevr':
    dset = build_dset_withpairs(args, checkpoint, vocab_t)
    collate_fn = collate_fn_withpairs

  loader_kwargs = {
    'batch_size': 1,
    'num_workers': args.loader_num_workers,
    'shuffle': args.shuffle,
    'collate_fn': collate_fn,
  }
  loader = DataLoader(dset, **loader_kwargs)

  return loader

def build_coco_dsets(args):
    dset_kwargs = {
      'image_dir': args.coco_train_image_dir,
      'instances_json': args.coco_train_instances_json,
      'stuff_json': args.coco_train_stuff_json,
      'stuff_only': args.coco_stuff_only,
      'image_size': args.image_size,
      'mask_size': args.mask_size,
      'max_samples': args.num_train_samples,
      'min_object_size': args.min_object_size,
      'min_objects_per_image': args.min_objects_per_image,
      'instance_whitelist': args.instance_whitelist,
      'stuff_whitelist': args.stuff_whitelist,
      'include_other': args.coco_include_other,
      'include_relationships': args.include_relationships,
      'no_mask': args.coco_no_mask,
    }
    train_dset = CocoSceneGraphDataset(**dset_kwargs)
    num_objs = train_dset.total_objects()
    num_imgs = len(train_dset)
    print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

    dset_kwargs['image_dir'] = args.coco_val_image_dir
    dset_kwargs['instances_json'] = args.coco_val_instances_json
    dset_kwargs['stuff_json'] = args.coco_val_stuff_json
    dset_kwargs['max_samples'] = args.num_val_samples
    val_dset = CocoSceneGraphDataset(**dset_kwargs)

    assert train_dset.vocab == val_dset.vocab
    vocab = json.loads(json.dumps(train_dset.vocab))

    return vocab, train_dset, val_dset

def build_train_dsets(args):
  print("building unpaired %s dataset" % args.dataset)
  with open(args.vocab_json, 'r') as f:
    vocab = json.load(f)
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.train_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.image_size,
    'max_samples': args.num_train_samples,
    'max_objects': args.max_objects_per_image,
    'use_orphaned_objects': args.vg_use_orphaned_objects,
    'include_relationships': args.include_relationships,
  }
  train_dset = SceneGraphNoPairsDataset(**dset_kwargs)
  iter_per_epoch = len(train_dset) // args.batch_size
  print('There are %d iterations per epoch' % iter_per_epoch)

  dset_kwargs['h5_path'] = args.val_h5
  del dset_kwargs['max_samples']
  val_dset = SceneGraphNoPairsDataset(**dset_kwargs)

  return vocab, train_dset, val_dset


def build_train_loaders(args):

  print(args.dataset)
  if args.dataset == 'vg' or (args.dataset == "clevr" and not args.is_supervised):
    vocab, train_dset, val_dset = build_train_dsets(args)
    collate_fn = collate_fn_nopairs

  elif args.dataset == 'coco':
    vocab, train_dset, val_dset = build_coco_dsets(args)
    collate_fn = coco_collate_fn
  elif args.dataset == 'clevr':
    vocab, train_dset, val_dset, test_dset = build_clevr_supervised_train_dsets(args)
    collate_fn = collate_fn_withpairs

  loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': True,
    'collate_fn': collate_fn,
  }
  train_loader = DataLoader(train_dset, **loader_kwargs)

  loader_kwargs['shuffle'] = args.shuffle_val
  val_loader = DataLoader(val_dset, **loader_kwargs)

  return vocab, train_loader, val_loader
