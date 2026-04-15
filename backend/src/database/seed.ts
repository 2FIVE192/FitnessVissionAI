import { Pool } from 'pg';

// standalone seed script — run with: npm run seed

const connectionString =
  process.env.DATABASE_URL ||
  'postgresql://fitness:fitness@localhost:5435/fitness_db';

const pool = new Pool({ connectionString });

async function seed() {
  const client = await pool.connect();
  try {
    // skip if data already exists
    const existing = await client.query('SELECT COUNT(*) FROM workouts');
    const alreadySeeded = Number(existing.rows[0].count) > 0;

const exerciseRenames: [string, string][] = [
  // English → Russian
  ['Push-ups',                          'Отжимания'],
  ['Wide Push-ups',                     'Отжимания широким хватом'],
  ['Diamond Push-ups',                  'Алмазные отжимания'],
  ['Chest Dips (bench)',                'Обратные отжимания на скамье'],
  ['Superman Hold',                     'Удержание «Супермен»'],
  ['Reverse Snow Angels',               'Обратные снежные ангелы'],
  ['Prone Y Raises',                    'Подъёмы рук лёжа в позе Y'],
  ['Bicep Curl (bodyweight)',           'Сгибание рук на бицепс с собственным весом'],
  ['Tricep Dips',                       'Отжимания на трицепс'],
  ['Arm Circles',                       'Круговые вращения руками'],
  ['Crunches',                          'Скручивания'],
  ['Plank',                             'Планка'],
  ['Mountain Climbers',                 'Альпинист'],
  ['Leg Raises',                        'Подъёмы ног'],
  ['Bicycle Crunches',                  'Велосипедные скручивания'],
  ['Glute Bridge',                      'Ягодичный мостик'],
  ['Donkey Kicks',                      'Махи ногой назад'],
  ['Fire Hydrants',                     'Отведение ноги в сторону на четвереньках'],
  ['Squats',                            'Приседания'],
  ['Lunges',                            'Выпады'],
  ['Wall Sit',                          'Стульчик у стены'],
  ['Calf Raises',                       'Подъёмы на носки'],
  ['Burpees',                           'Бёрпи'],
  ['Jumping Jacks',                     'Прыжки с разведением рук и ног'],
  ['High Knees',                        'Бег с высоким подниманием коленей'],
  ['Jump Squats',                       'Прыжковые приседания'],
  ['Forward Fold',                      'Наклон вперёд'],
  ['Cat-Cow Stretch',                   'Кошка-корова'],
  ["Child's Pose",                      'Поза ребёнка'],
  ['Downward Dog',                      'Собака мордой вниз'],
  ['Warrior I',                         'Воин I'],
  ['Warrior II',                        'Воин II'],
  ['Tree Pose',                         'Поза дерева'],
  ['Широкие отжимания',                 'Отжимания широким хватом'],
  ['Отжимания «алмаз»',                 'Алмазные отжимания'],
  ['Отжимания на скамье',               'Обратные отжимания на скамье'],
  ['Упражнение «Супермен»',             'Удержание «Супермен»'],
  ['Подъёмы рук в позе Y',              'Подъёмы рук лёжа в позе Y'],
  ['Сгибания бицепса',                  'Сгибание рук на бицепс с собственным весом'],
  ['Разгибания на трицепс',             'Отжимания на трицепс'],
  ['Вращение руками',                   'Круговые вращения руками'],
  ['Бег в планке',                      'Альпинист'],
  ['Велосипед',                         'Велосипедные скручивания'],
  ['Ягодичный мост',                    'Ягодичный мостик'],
  ['Отведение ноги в сторону',          'Отведение ноги в сторону на четвереньках'],
  ['Прыжки с разведением',              'Прыжки с разведением рук и ног'],
  ['Бег с высоким подниманием колен',   'Бег с высоким подниманием коленей'],
];

const workoutRenames: [string, string][] = [
  // English → Russian
  ['Chest Blast',                'Грудь'],
  ['Arm Toner',                  'Тонус рук'],
  ['Leg Day',                    'День ног'],
  ['Back Strength',              'Силовая тренировка спины'],
  ['Glute Builder',              'Ягодицы'],
  ['Quick Cardio Burn',          'Быстрое кардио'],
  ['Fat Burner Express',         'Экспресс-жиросжигание'],
  ['HIIT Core Crusher',          'ВИИТ: пресс и мышцы кора'],
  ['Full Body HIIT',             'ВИИТ на всё тело'],
  ['Morning Stretch',            'Утренняя растяжка'],
  ['Yoga Flow',                  'Плавная йога'],
  ['Core Abs Blast',             'Пресс и мышцы кора'],
  ['Проработка груди',           'Грудь'],
  ['Сила спины',                 'Силовая тренировка спины'],
  ['Жиросжигающая тренировка',   'Экспресс-жиросжигание'],
  ['ВИИТ: пресс и кор',          'ВИИТ: пресс и мышцы кора'],
  ['ВИИТ: всё тело',             'ВИИТ на всё тело'],
  ['Йога-флоу',                  'Плавная йога'],
  ['Пресс и кор',                'Пресс и мышцы кора'],
];

    if (alreadySeeded) {
      console.log('database already seeded — updating names to Russian...');
      for (const [oldTitle, newTitle] of exerciseRenames) {
        await client.query('UPDATE exercises SET title = $1 WHERE title = $2', [newTitle, oldTitle]);
      }
      for (const [oldTitle, newTitle] of workoutRenames) {
        await client.query('UPDATE workouts SET title = $1 WHERE title = $2', [newTitle, oldTitle]);
      }
      console.log('names updated.');
      return;
    }

    console.log('seeding database...');

    // -- exercises --
    const exercises = [
      { title: 'Отжимания',                    muscle_group: 'CHEST',     met: 8.0,  duration_seconds: 60, rest_after_seconds: 15 },
      { title: 'Широкие отжимания',            muscle_group: 'CHEST',     met: 8.0,  duration_seconds: 45, rest_after_seconds: 15 },
      { title: 'Отжимания «алмаз»',            muscle_group: 'CHEST',     met: 9.0,  duration_seconds: 45, rest_after_seconds: 20 },
      { title: 'Отжимания на скамье',          muscle_group: 'CHEST',     met: 8.5,  duration_seconds: 60, rest_after_seconds: 15 },
      { title: 'Упражнение «Супермен»',        muscle_group: 'BACK',      met: 4.0,  duration_seconds: 30, rest_after_seconds: 15 },
      { title: 'Обратные снежные ангелы',      muscle_group: 'BACK',      met: 5.0,  duration_seconds: 45, rest_after_seconds: 15 },
      { title: 'Подъёмы рук в позе Y',         muscle_group: 'BACK',      met: 4.5,  duration_seconds: 40, rest_after_seconds: 10 },
      { title: 'Сгибания бицепса',             muscle_group: 'ARMS',      met: 5.0,  duration_seconds: 45, rest_after_seconds: 10 },
      { title: 'Разгибания на трицепс',        muscle_group: 'ARMS',      met: 7.0,  duration_seconds: 45, rest_after_seconds: 15 },
      { title: 'Вращение руками',              muscle_group: 'ARMS',      met: 3.5,  duration_seconds: 30, rest_after_seconds: 10 },
      { title: 'Скручивания',                  muscle_group: 'ABS',       met: 5.0,  duration_seconds: 60, rest_after_seconds: 15 },
      { title: 'Планка',                       muscle_group: 'ABS',       met: 4.0,  duration_seconds: 45, rest_after_seconds: 15 },
      { title: 'Бег в планке',                 muscle_group: 'ABS',       met: 8.0,  duration_seconds: 30, rest_after_seconds: 15 },
      { title: 'Подъёмы ног',                  muscle_group: 'ABS',       met: 5.0,  duration_seconds: 45, rest_after_seconds: 15 },
      { title: 'Велосипед',                    muscle_group: 'ABS',       met: 6.0,  duration_seconds: 45, rest_after_seconds: 15 },
      { title: 'Ягодичный мост',               muscle_group: 'GLUTES',    met: 5.0,  duration_seconds: 45, rest_after_seconds: 10 },
      { title: 'Махи ногой назад',             muscle_group: 'GLUTES',    met: 5.5,  duration_seconds: 40, rest_after_seconds: 10 },
      { title: 'Отведение ноги в сторону',     muscle_group: 'GLUTES',    met: 5.0,  duration_seconds: 40, rest_after_seconds: 10 },
      { title: 'Приседания',                   muscle_group: 'LEGS',      met: 7.0,  duration_seconds: 60, rest_after_seconds: 15 },
      { title: 'Выпады',                       muscle_group: 'LEGS',      met: 7.0,  duration_seconds: 60, rest_after_seconds: 15 },
      { title: 'Стульчик у стены',             muscle_group: 'LEGS',      met: 4.0,  duration_seconds: 45, rest_after_seconds: 15 },
      { title: 'Подъёмы на носки',             muscle_group: 'LEGS',      met: 4.0,  duration_seconds: 30, rest_after_seconds: 10 },
      { title: 'Бёрпи',                        muscle_group: 'FULL_BODY', met: 10.0, duration_seconds: 30, rest_after_seconds: 20 },
      { title: 'Прыжки с разведением',         muscle_group: 'FULL_BODY', met: 8.0,  duration_seconds: 60, rest_after_seconds: 15 },
      { title: 'Бег с высоким подниманием колен', muscle_group: 'FULL_BODY', met: 9.0, duration_seconds: 30, rest_after_seconds: 15 },
      { title: 'Прыжковые приседания',         muscle_group: 'FULL_BODY', met: 9.5,  duration_seconds: 30, rest_after_seconds: 20 },
      { title: 'Наклон вперёд',               muscle_group: 'LEGS',      met: 2.5,  duration_seconds: 30, rest_after_seconds: 5 },
      { title: 'Кошка-корова',                 muscle_group: 'BACK',      met: 2.0,  duration_seconds: 30, rest_after_seconds: 5 },
      { title: 'Поза ребёнка',                 muscle_group: 'BACK',      met: 2.0,  duration_seconds: 30, rest_after_seconds: 5 },
      { title: 'Собака мордой вниз',           muscle_group: 'FULL_BODY', met: 3.0,  duration_seconds: 30, rest_after_seconds: 5 },
      { title: 'Воин I',                       muscle_group: 'LEGS',      met: 3.0,  duration_seconds: 30, rest_after_seconds: 5 },
      { title: 'Воин II',                      muscle_group: 'LEGS',      met: 3.0,  duration_seconds: 30, rest_after_seconds: 5 },
      { title: 'Поза дерева',                  muscle_group: 'LEGS',      met: 2.5,  duration_seconds: 30, rest_after_seconds: 5 },
    ];

    const exerciseIds: Record<string, string> = {};
    for (const e of exercises) {
      const res = await client.query(
        `INSERT INTO exercises (title, muscle_group, met, duration_seconds, rest_after_seconds)
         VALUES ($1, $2, $3, $4, $5) RETURNING id`,
        [e.title, e.muscle_group, e.met, e.duration_seconds, e.rest_after_seconds],
      );
      exerciseIds[e.title] = res.rows[0].id;
    }
    console.log(`  ${exercises.length} exercises inserted`);

    // -- workouts --
    const workouts = [
      {
        title: 'Проработка груди', type: 'STRENGTH', difficulty: 'MEDIUM',
        duration_minutes: 12, is_recommended: true,
        exercises: ['Отжимания', 'Широкие отжимания', 'Отжимания «алмаз»', 'Отжимания на скамье'],
      },
      {
        title: 'Тонус рук', type: 'STRENGTH', difficulty: 'EASY',
        duration_minutes: 8, is_recommended: false,
        exercises: ['Сгибания бицепса', 'Разгибания на трицепс', 'Вращение руками'],
      },
      {
        title: 'День ног', type: 'STRENGTH', difficulty: 'HARD',
        duration_minutes: 15, is_recommended: true,
        exercises: ['Приседания', 'Выпады', 'Стульчик у стены', 'Подъёмы на носки', 'Прыжковые приседания'],
      },
      {
        title: 'Сила спины', type: 'STRENGTH', difficulty: 'MEDIUM',
        duration_minutes: 10, is_recommended: false,
        exercises: ['Упражнение «Супермен»', 'Обратные снежные ангелы', 'Подъёмы рук в позе Y'],
      },
      {
        title: 'Ягодицы', type: 'STRENGTH', difficulty: 'EASY',
        duration_minutes: 8, is_recommended: false,
        exercises: ['Ягодичный мост', 'Махи ногой назад', 'Отведение ноги в сторону'],
      },
      {
        title: 'Быстрое кардио', type: 'CARDIO', difficulty: 'MEDIUM',
        duration_minutes: 10, is_recommended: true,
        exercises: ['Прыжки с разведением', 'Бег с высоким подниманием колен', 'Бёрпи', 'Бег в планке'],
      },
      {
        title: 'Жиросжигающая тренировка', type: 'CARDIO', difficulty: 'HARD',
        duration_minutes: 12, is_recommended: true,
        exercises: ['Бёрпи', 'Прыжковые приседания', 'Бег с высоким подниманием колен', 'Бег в планке', 'Прыжки с разведением'],
      },
      {
        title: 'ВИИТ: пресс и кор', type: 'HIIT', difficulty: 'HARD',
        duration_minutes: 15, is_recommended: true,
        exercises: ['Бёрпи', 'Бег в планке', 'Велосипед', 'Планка', 'Прыжковые приседания'],
      },
      {
        title: 'ВИИТ: всё тело', type: 'HIIT', difficulty: 'MEDIUM',
        duration_minutes: 15, is_recommended: true,
        exercises: ['Прыжки с разведением', 'Приседания', 'Отжимания', 'Скручивания', 'Бег с высоким подниманием колен'],
      },
      {
        title: 'Утренняя растяжка', type: 'STRETCHING', difficulty: 'EASY',
        duration_minutes: 7, is_recommended: true,
        exercises: ['Наклон вперёд', 'Кошка-корова', 'Поза ребёнка', 'Собака мордой вниз'],
      },
      {
        title: 'Йога-флоу', type: 'YOGA', difficulty: 'EASY',
        duration_minutes: 10, is_recommended: true,
        exercises: ['Собака мордой вниз', 'Воин I', 'Воин II', 'Поза дерева', 'Поза ребёнка'],
      },
      {
        title: 'Пресс и кор', type: 'STRENGTH', difficulty: 'MEDIUM',
        duration_minutes: 12, is_recommended: false,
        exercises: ['Скручивания', 'Планка', 'Бег в планке', 'Подъёмы ног', 'Велосипед'],
      },
    ];

    for (const w of workouts) {
      const res = await client.query(
        `INSERT INTO workouts (title, type, difficulty, duration_minutes, is_recommended)
         VALUES ($1, $2, $3, $4, $5) RETURNING id`,
        [w.title, w.type, w.difficulty, w.duration_minutes, w.is_recommended],
      );
      const workoutId = res.rows[0].id;

      for (let i = 0; i < w.exercises.length; i++) {
        const exId = exerciseIds[w.exercises[i]];
        if (!exId) continue;
        await client.query(
          `INSERT INTO workout_exercises (workout_id, exercise_id, sort_order)
           VALUES ($1, $2, $3)`,
          [workoutId, exId, i],
        );
      }
    }
    console.log(`  ${workouts.length} workouts inserted`);
    console.log('seeding complete.');
  } finally {
    client.release();
    await pool.end();
  }
}

seed().catch((err) => {
  console.error('seeding failed:', err);
  process.exit(1);
});
