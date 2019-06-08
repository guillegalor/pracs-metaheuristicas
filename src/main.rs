extern crate csv;
#[macro_use]
extern crate prettytable;

use std::env;

use rand::distributions::{Distribution, Normal, Uniform};
use rand::prelude::*;
use rand::seq::SliceRandom;

use std::collections::HashMap;
use std::error::Error;
use std::process;
use std::time::Instant;

use std::cmp::Ordering;

// ---------------------------------------------------------------------------------
// Auxiliary functions
// ---------------------------------------------------------------------------------
fn truncate_01(n: &f32) -> f32 {
    if *n < 0. {
        0.
    } else if *n > 1. {
        1.
    } else {
        *n
    }
}

// ---------------------------------------------------------------------------------
// DataElem definitions and specific functions for it
// ---------------------------------------------------------------------------------

type Chromosome = Vec<f32>;

#[derive(Clone)]
pub struct ChromosomeAndResult {
    chromosome: Chromosome,
    result: f32,
}

impl ChromosomeAndResult {
    fn new(chromosome: Chromosome, result: f32) -> ChromosomeAndResult {
        ChromosomeAndResult {
            chromosome: chromosome,
            result: result,
        }
    }
}

impl PartialEq for ChromosomeAndResult {
    fn eq(&self, other: &ChromosomeAndResult) -> bool {
        self.result == other.result
    }
}

impl Eq for ChromosomeAndResult {}

impl Ord for ChromosomeAndResult {
    fn cmp(&self, other: &ChromosomeAndResult) -> Ordering {
        if self.result < other.result {
            return Ordering::Less;
        } else if self.result > other.result {
            return Ordering::Greater;
        } else {
            return Ordering::Equal;
        }
    }
}

impl PartialOrd for ChromosomeAndResult {
    fn partial_cmp(&self, other: &ChromosomeAndResult) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Trait for CSV data
///
/// **Note**: The struct implementing this trait must also implement `Copy` and `Clone`.

pub trait DataElem<T> {
    fn new() -> T;
    fn get_num_attributes() -> usize;
    fn get_id(&self) -> i32;
    fn get_class(&self) -> i32;
    fn get_attribute(&self, index: usize) -> f32;
    fn set_id(&mut self, id: i32);
    fn set_class(&mut self, class: i32);
    fn set_attribute(&mut self, index: usize, attr: f32);
}

/// Calculate the "euclidian_distance" distance between two `T` elements
/// asigning a weight to every attribute
///
/// # Arguments
///
/// * `one` - First `T` element
/// * `other` - Second `T` element
/// * `weights` - Vec with a weight(between 0 and 1) for every attribute
///
/// # Returns
/// A `Result ` with:
/// * `Err(&'static str)` if the number or weights doesn't coincide with the number ot attrs of `T`
/// * `Ok(f32)`
pub fn eu_dist_with_weigths<T: DataElem<T> + Copy + Clone>(
    one: &T,
    other: &T,
    weights: &Vec<f32>,
) -> Result<f32, &'static str> {
    if T::get_num_attributes() != weights.len() {
        return Err("El numero de pesos no coincide con el numero de atributos");
    }

    let mut dist = 0.0;
    for attr in 0..T::get_num_attributes() {
        dist += (one.get_attribute(attr) - other.get_attribute(attr))
            * (one.get_attribute(attr) - other.get_attribute(attr))
            * weights[attr];
    }

    dist = dist.sqrt();

    return Ok(dist);
}

// ---------------------------------------------------------------------------------
// Structs for every data set
// ---------------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct TextureRecord {
    id: i32,
    attributes: [f32; 40],
    class: i32,
}

#[derive(Copy, Clone)]
struct ColposcopyRecord {
    id: i32,
    attributes: [f32; 62],
    class: i32,
}

#[derive(Copy, Clone)]
struct IonosphereRecord {
    id: i32,
    attributes: [f32; 34],
    class: i32,
}

impl DataElem<TextureRecord> for TextureRecord {
    fn new() -> TextureRecord {
        TextureRecord {
            id: -1,
            attributes: [0.0; 40],
            class: -1,
        }
    }

    fn get_num_attributes() -> usize {
        return 40;
    }

    fn get_id(&self) -> i32 {
        return self.id;
    }

    fn get_class(&self) -> i32 {
        return self.class;
    }

    // Precondition: index is in range
    fn get_attribute(&self, index: usize) -> f32 {
        return self.attributes[index];
    }

    fn set_id(&mut self, id: i32) {
        self.id = id;
    }

    fn set_class(&mut self, class: i32) {
        self.class = class;
    }

    fn set_attribute(&mut self, index: usize, attr: f32) {
        self.attributes[index] = attr;
    }
}

impl DataElem<ColposcopyRecord> for ColposcopyRecord {
    fn new() -> ColposcopyRecord {
        ColposcopyRecord {
            id: -1,
            attributes: [0.0; 62],
            class: -1,
        }
    }

    fn get_num_attributes() -> usize {
        return 62;
    }

    fn get_id(&self) -> i32 {
        return self.id;
    }

    fn get_class(&self) -> i32 {
        return self.class;
    }

    // Precondition: index is in range
    fn get_attribute(&self, index: usize) -> f32 {
        return self.attributes[index];
    }

    fn set_id(&mut self, id: i32) {
        self.id = id;
    }

    fn set_class(&mut self, class: i32) {
        self.class = class;
    }

    fn set_attribute(&mut self, index: usize, attr: f32) {
        self.attributes[index] = attr;
    }
}

impl DataElem<IonosphereRecord> for IonosphereRecord {
    fn new() -> IonosphereRecord {
        IonosphereRecord {
            id: -1,
            attributes: [0.0; 34],
            class: -1,
        }
    }

    fn get_num_attributes() -> usize {
        return 34;
    }

    fn get_id(&self) -> i32 {
        return self.id;
    }

    fn get_class(&self) -> i32 {
        return self.class;
    }

    // Precondition: index is in range
    fn get_attribute(&self, index: usize) -> f32 {
        return self.attributes[index];
    }

    fn set_id(&mut self, id: i32) {
        self.id = id;
    }

    fn set_class(&mut self, class: i32) {
        self.class = class;
    }

    fn set_attribute(&mut self, index: usize, attr: f32) {
        self.attributes[index] = attr;
    }
}

//---------------------------------------------------------------------------------------
// Auxiliary functions to treat data set
//---------------------------------------------------------------------------------------

/// Normalize attributes of a vec of `T` so everyone is in [0,1]
/// # Arguments
///
/// * `data` - Vec of `T` being normalized
///
/// # Returns
/// Returns a vec of `T` normalized

pub fn normalize_data<T: DataElem<T> + Copy + Clone>(data: Vec<T>) -> Vec<T> {
    let num_attrs = T::get_num_attributes();

    let mut mins = vec![std::f32::MAX; num_attrs];
    let mut maxs = vec![std::f32::MIN; num_attrs];

    for elem in data.iter() {
        for attr in 0..num_attrs {
            // Calculates min
            if elem.get_attribute(attr) < mins[attr] {
                mins[attr] = elem.get_attribute(attr);
            }
            // Calculates max
            if elem.get_attribute(attr) > maxs[attr] {
                maxs[attr] = elem.get_attribute(attr);
            }
        }
    }

    let mut new_data = data;

    let mut max_distances = vec![0.0; num_attrs];
    for attr in 0..num_attrs {
        max_distances[attr] = maxs[attr] - mins[attr];
    }

    for elem in new_data.iter_mut() {
        for attr in 0..num_attrs {
            if max_distances[attr] != 0.0 {
                elem.set_attribute(
                    attr,
                    (elem.get_attribute(attr) - mins[attr]) / max_distances[attr],
                );
            }
        }
    }

    return new_data;
}

/// Makes partitions keeping the class diversity.
///
/// # Arguments
///
/// * `data` - Vec with all the data.
///
/// # Returns
/// Returns a vector with 5 vectors of `T`.
pub fn make_partitions<T: DataElem<T> + Copy + Clone>(data: &Vec<T>) -> Vec<Vec<T>> {
    let folds = 5;

    // Hash tab that has an entry for every diferent class, and tells where should be
    // the next elemt
    let mut categories_count = HashMap::new();
    let mut partitions: Vec<Vec<T>> = Vec::new();

    for _ in 0..folds {
        partitions.push(Vec::new());
    }

    for example in data.iter() {
        // If that class was already found, gets the partition that this elem should go
        // If not, insert a new entry for that class with value 0 (first partition)
        let counter = categories_count.entry(example.get_class()).or_insert(0);

        partitions[*counter].push(example.clone());

        // The next elem with same class as elem will go to the next partition
        *counter = (*counter + 1) % folds;
    }

    return partitions;
}

//---------------------------------------------------------------------------------------
// Core functions for algorithms
//---------------------------------------------------------------------------------------

/// Classifies one `T` item using a Vec of `T` elems, a Vec of weights and the
/// 1nn algorithm
///
/// # Arguments
///
/// * `data` - Vec of `T` used as knowledge
/// * `item` - `T` elem being classified
/// * `weights` - Vec of f32 representing weights
///
/// # Returns
/// A `Result` with:
/// * `Err(&'static str)` if the number or weights doesn't coincide with the number ot attrs of `T`
/// * `Ok(i32)`

pub fn classifier_1nn_with_weights<T: DataElem<T> + Copy + Clone>(
    data: &Vec<T>,
    item: &T,
    weights: &Vec<f32>,
) -> Result<i32, Box<Error>> {
    let mut c_min = data[0].get_class();
    let mut d_min = eu_dist_with_weigths(item, &data[0], weights)?;
    let weights_discarding: Vec<f32> = weights
        .iter()
        .map(|x| if *x < 0.2 { 0.0 } else { *x })
        .collect();

    for example in data.iter().skip(1) {
        if example.get_id() != item.get_id() {
            let aux_distance = eu_dist_with_weigths(item, example, &weights_discarding)?;
            if aux_distance < d_min {
                c_min = example.get_class();
                d_min = aux_distance;
            }
        }
    }

    return Ok(c_min);
}

/// Calculates a vec of weights using a vec of `T` elems as knowledge
///
/// # Arguments
///
/// * `data` - Vec of `T` used as knowledge
///
/// # Returns
/// A Vec with the same num of f32 elems as the num of attributes of `T`

pub fn relief_algorithm<T: DataElem<T> + Copy + Clone>(data: &Vec<T>) -> Vec<f32> {
    let num_attrs = T::get_num_attributes();
    let mut weights = vec![0.0; num_attrs];
    let weights_eu_dist = vec![1.0; num_attrs];

    for elem in data.iter() {
        let mut nearest_enemy_index = 0;
        let mut nearest_ally_index = 0;
        let mut best_enemy_distance = std::f32::MAX;
        let mut best_ally_distance = std::f32::MAX;

        for (index, candidate) in data.iter().enumerate() {
            if elem.get_id() != candidate.get_id() {
                let aux_distance = eu_dist_with_weigths(elem, candidate, &weights_eu_dist)
                    .expect("Euclidian distance with weights relief algorithm");

                // Ally search
                if elem.get_class() == candidate.get_class() {
                    if aux_distance < best_ally_distance {
                        best_ally_distance = aux_distance;
                        nearest_ally_index = index;
                    }
                }
                // Enemy search
                else {
                    if aux_distance < best_enemy_distance {
                        best_enemy_distance = aux_distance;
                        nearest_enemy_index = index;
                    }
                }
            }
        }

        let nearest_ally = data[nearest_ally_index].clone();
        let nearest_enemy = data[nearest_enemy_index].clone();

        for attr in 0..num_attrs {
            let attr_ally_dist =
                (elem.get_attribute(attr) - nearest_ally.get_attribute(attr)).abs();
            let attr_enemy_dist =
                (elem.get_attribute(attr) - nearest_enemy.get_attribute(attr)).abs();

            weights[attr] += attr_enemy_dist - attr_ally_dist;
        }
    }

    let mut max_weight = weights[0];

    for w in weights.iter() {
        if *w > max_weight {
            max_weight = *w;
        }
    }

    for w in weights.iter_mut() {
        if *w < 0.0 {
            *w = 0.0;
        } else {
            *w = *w / max_weight;
        }
    }

    return weights;
}

// TODO Move this to its place
pub fn random_weights_generator(num_attrs: usize, rng: &mut StdRng) -> Vec<f32> {
    // Normal distribution with mean = 0.0, standard deviation = 0.3
    let uniform = Uniform::new(0.0, 1.0);

    // Initialize random weights (using normal distribution)
    let mut weights: Vec<f32> = Vec::with_capacity(num_attrs);
    for _ in 0..num_attrs {
        weights.push(uniform.sample(rng));
    }

    weights
}

// Local search
// Return the weights vec and the evaluation rate
pub fn local_search<T: DataElem<T> + Copy + Clone>(
    data: &Vec<T>,
    rng: &mut StdRng,
    initial_weights: &Vec<f32>,
    mutation_operator: fn(&mut Vec<f32>, usize, &mut StdRng),
    max_mutations: usize,
    max_neighbour_without_muting: usize,
) -> Vec<f32> {
    let num_attrs = T::get_num_attributes();

    // Closure that fills and shuffle indices
    let fill_and_shuffle = |indices: &mut Vec<usize>, rng: &mut StdRng| {
        *indices = (0..num_attrs).collect();
        indices.shuffle(rng);
    };

    // Initialize vector of indices and shuffles it
    let mut indices: Vec<usize> = Vec::with_capacity(num_attrs);
    fill_and_shuffle(&mut indices, rng);

    // Local search parameters
    let mut num_of_mutations = 0;
    let mut neighbours_without_mutting = 0;

    // Initialize weights
    let mut weights: Vec<f32> = initial_weights.clone();

    let mut current_ev_rate = evaluate(data, data, &weights, 0.5).2;

    while neighbours_without_mutting < max_neighbour_without_muting
        && num_of_mutations < max_mutations
    {
        let mut aux_weights = weights.clone();

        let index = indices.pop().expect("El vector está vacio");

        // Mutation
        mutation_operator(&mut aux_weights, index, rng);

        // Evaluation
        let aux_ev_rate = evaluate(data, data, &aux_weights, 0.5).2;

        if aux_ev_rate > current_ev_rate {
            current_ev_rate = aux_ev_rate;
            weights = aux_weights;

            neighbours_without_mutting = 0;

            // Refreshes indices if improves
            fill_and_shuffle(&mut indices, rng);
        } else {
            neighbours_without_mutting += 1;

            if indices.is_empty() {
                fill_and_shuffle(&mut indices, rng);
            }
        }

        num_of_mutations += 1;
    }
    return weights;
}

pub fn generational_genetic_algorithm<T: DataElem<T> + Copy + Clone>(
    data: &Vec<T>,
    rng: &mut StdRng,
    population_size: usize,
    crossover_probability: f32,
    mutation_probability: f32,
    max_calls_to_eval: usize,
    selection_operator: fn(
        &Vec<ChromosomeAndResult>,
        usize,
        &mut StdRng,
    ) -> Vec<ChromosomeAndResult>,
    crossover_operator: fn(&Chromosome, &Chromosome, &mut StdRng) -> (Chromosome, Chromosome),
    mutation_operator: fn(&mut Vec<f32>, usize, &mut StdRng),
) -> Vec<f32> {
    const DEBUG: bool = false;

    let quick_eval = |weights: &Chromosome| {
        return evaluate(data, data, weights, 0.5).2;
    };

    let mut generation_counter = 0;

    let num_gens_per_chromosome = T::get_num_attributes();
    let uniform = Uniform::new(0.0, 1.0);
    let mut calls_to_eval = 0;

    // Initialization ----
    let mut current_population: Vec<ChromosomeAndResult> = Vec::with_capacity(population_size);

    let mut aux_weights: Vec<f32> = Vec::with_capacity(T::get_num_attributes());
    // Initialize random weights (using normal distribution)
    for _ in 0..population_size {
        for _ in 0..num_gens_per_chromosome {
            aux_weights.push(uniform.sample(rng));
        }

        let aux_chrom_and_result = ChromosomeAndResult {
            chromosome: aux_weights.clone(),
            result: quick_eval(&aux_weights),
        };

        calls_to_eval += 1;

        current_population.push(aux_chrom_and_result);

        aux_weights.clear();
    }

    current_population.sort();
    let mut current_best_chrom_res = current_population.last().unwrap().clone();

    while calls_to_eval < max_calls_to_eval {
        if false {
            println!("Generación número: {}", generation_counter);
            println!("Llamadas a eval: {}", calls_to_eval);
            println!("Mejor cromosoma: {}", current_best_chrom_res.result);
        }

        if DEBUG {
            for elem in current_population.iter() {
                println!("{}", elem.result);
            }
            println!("\n");
        }

        let mut auxiliar_population: Vec<ChromosomeAndResult> = Vec::with_capacity(population_size);

        // Selection  ----
        let parents = selection_operator(&current_population, population_size, rng);

        // Crossover (arithmetic mean) ----
        let num_of_crossovers = (crossover_probability * (population_size as f32) / 2.) as usize;

        if DEBUG {
            println!("Num of crossover: {}", num_of_crossovers);
            println!("Parents");
            for elem in parents.iter() {
                println!("{}", elem.result);
            }
        }

        // Adds children chromosomes
        for index in 0..num_of_crossovers {
            let children = crossover_operator(
                &parents[2 * index].chromosome,
                &parents[2 * index + 1].chromosome,
                rng,
            );

            auxiliar_population.push(ChromosomeAndResult::new(children.0, -1.));
            auxiliar_population.push(ChromosomeAndResult::new(children.1, -1.));
        }

        // Adds remaining chromosomes from parents
        for index in 2 * num_of_crossovers..population_size {
            auxiliar_population.push(parents[index].clone());
        }

        if DEBUG {
            println!("Childrens + Some Parents");
            for elem in auxiliar_population.iter() {
                println!("{}", elem.result);
            }
        }

        // Mutation ----
        let expected_num_of_mutations =
            mutation_probability * population_size as f32 * num_gens_per_chromosome as f32;
        let mut num_of_mutations = expected_num_of_mutations as usize;

        if rng.gen_range(0., 1.) < (expected_num_of_mutations - num_of_mutations as f32) {
            num_of_mutations += 1;
        }

        if DEBUG {
            println!("Num of mutations: {}", num_of_mutations);
        }

        for _ in 0..num_of_mutations {
            let selector = rng.gen_range(0, population_size * num_gens_per_chromosome);
            let chosen_chromosome = &mut auxiliar_population[selector % population_size].chromosome;
            let chosen_gene = selector / population_size;

            // Mutation operator
            mutation_operator(chosen_chromosome, chosen_gene, rng);

            auxiliar_population[selector % population_size].result = quick_eval(chosen_chromosome);
            calls_to_eval += 1;
        }

        // Evaluation ----
        for chrom_and_res in auxiliar_population.iter_mut() {
            if chrom_and_res.result == -1. {
                chrom_and_res.result = quick_eval(&chrom_and_res.chromosome);
                calls_to_eval += 1;
            }
        }

        auxiliar_population.sort();
        let new_best_chrom_res = auxiliar_population.last().unwrap().clone();

        if DEBUG {
            println!("New population(without replacing)");
            for elem in auxiliar_population.iter() {
                println!("{}", elem.result);
            }
        }

        // Replacement ----
        // If new best chrom is worst than the old one
        if new_best_chrom_res < current_best_chrom_res {
            auxiliar_population.remove(0);
            auxiliar_population.push(current_best_chrom_res.clone());
        } else {
            current_best_chrom_res = new_best_chrom_res;
        }

        current_population = auxiliar_population;

        generation_counter += 1;
    }

    return current_best_chrom_res.chromosome;
}

pub fn steady_state_genetic_algorithm<T: DataElem<T> + Copy + Clone>(
    data: &Vec<T>,
    rng: &mut StdRng,
    population_size: usize,
    mutation_probability: f32,
    max_calls_to_eval: usize,
    selection_operator: fn(
        &Vec<ChromosomeAndResult>,
        usize,
        &mut StdRng,
    ) -> Vec<ChromosomeAndResult>,
    crossover_operator: fn(&Chromosome, &Chromosome, &mut StdRng) -> (Chromosome, Chromosome),
    mutation_operator: fn(&mut Vec<f32>, usize, &mut StdRng),
) -> Vec<f32> {
    const DEBUG: bool = false;

    let quick_eval = |weights: &Chromosome| {
        return evaluate(data, data, weights, 0.5).2;
    };

    let mut generation_counter = 0;

    let num_gens_per_chromosome = T::get_num_attributes();
    let uniform = Uniform::new(0.0, 1.0);
    let mut calls_to_eval = 0;

    // Initialization ----
    let mut current_population: Vec<ChromosomeAndResult> = Vec::with_capacity(population_size);

    let mut aux_weights: Vec<f32> = Vec::with_capacity(T::get_num_attributes());
    // Initialize random weights (using normal distribution)
    for _ in 0..population_size {
        for _ in 0..num_gens_per_chromosome {
            aux_weights.push(uniform.sample(rng));
        }

        let aux_chrom_and_result = ChromosomeAndResult {
            chromosome: aux_weights.clone(),
            result: quick_eval(&aux_weights),
        };

        calls_to_eval += 1;

        current_population.push(aux_chrom_and_result);

        aux_weights.clear();
    }

    current_population.sort();
    let mut current_best_chrom_res = current_population.last().unwrap().clone();

    while calls_to_eval < max_calls_to_eval {
        if false {
            println!("Generación número: {}", generation_counter);
            println!("Llamadas a eval: {}", calls_to_eval);
            println!("Mejor cromosoma: {}", current_best_chrom_res.result);
        }

        if DEBUG {
            for elem in current_population.iter() {
                println!("{}", elem.result);
            }
            println!("\n");
        }

        // Selection  ----
        let parents = selection_operator(&current_population, 2, rng);

        // Crossover  ----
        let children = crossover_operator(&parents[0].chromosome, &parents[1].chromosome, rng);
        let mut auxiliar_population: Vec<ChromosomeAndResult> = Vec::with_capacity(2);
        auxiliar_population.push(ChromosomeAndResult::new(children.0.clone(), -1.));
        auxiliar_population.push(ChromosomeAndResult::new(children.1.clone(), -1.));

        // Mutation ----
        let expected_num_of_mutations = mutation_probability * 2. * num_gens_per_chromosome as f32;
        let mut num_of_mutations = expected_num_of_mutations as usize;

        if rng.gen_range(0., 1.) < (expected_num_of_mutations - num_of_mutations as f32) {
            num_of_mutations += 1;
        }

        for _ in 0..num_of_mutations {
            let selector = rng.gen_range(0, 2 * num_gens_per_chromosome);
            let chosen_chromosome = &mut auxiliar_population[selector % 2].chromosome;
            let chosen_gene = selector / 2;

            // Mutation operator
            mutation_operator(chosen_chromosome, chosen_gene, rng);

            auxiliar_population[selector % 2].result = quick_eval(chosen_chromosome);
            calls_to_eval += 1;
        }

        for chrom_and_res in auxiliar_population.iter_mut() {
            if chrom_and_res.result == -1. {
                chrom_and_res.result = quick_eval(&chrom_and_res.chromosome);
                calls_to_eval += 1;
            }
        }

        auxiliar_population.sort();

        // Replacement ----
        // Keeps two best of current_population[0], current_population[1],
        // auxiliar_population[0], auxiliar_population[1]

        // if previous best is worse than new worst replace both previous
        if current_population[1] < auxiliar_population[0] {
            current_population.remove(0);
            current_population.remove(1);
            current_population.push(auxiliar_population[0].clone());
            current_population.push(auxiliar_population[1].clone());
        }
        // if not, if previous worst is worse than new best replace only him
        else if current_population[0] < auxiliar_population[1] {
            current_population.remove(0);
            current_population.push(auxiliar_population[1].clone());
        }

        current_population.sort();
        current_best_chrom_res = current_population.last().unwrap().clone();

        generation_counter += 1;
    }

    return current_best_chrom_res.chromosome;
}

pub fn low_intensity_local_search<T: DataElem<T> + Copy + Clone>(
    data: &Vec<T>,
    weights_and_result: &mut ChromosomeAndResult,
    rng: &mut StdRng,
) -> usize {
    let num_attrs = T::get_num_attributes();
    let mut calls_to_eval = 0;

    // Closure that fills and shuffle indices
    let fill_and_shuffle = |indices: &mut Vec<usize>, rng: &mut StdRng| {
        *indices = (0..num_attrs).collect();
        indices.shuffle(rng);
    };

    // Initialize vector of indices and shuffles it
    let mut indices: Vec<usize> = Vec::with_capacity(num_attrs);
    fill_and_shuffle(&mut indices, rng);

    // Normal distribution with mean = 0.0, standard deviation = 0.3
    let normal = Normal::new(0.0, 0.3);

    for _ in 0..2 * num_attrs {
        let mut aux_weights = weights_and_result.chromosome.clone();

        let index = indices.pop().expect("El vector está vacio");

        // Mutation
        aux_weights[index] += normal.sample(rng) as f32;

        // Truncate into [0,1]
        if aux_weights[index] < 0. {
            aux_weights[index] = 0.;
        } else if aux_weights[index] > 1. {
            aux_weights[index] = 1.;
        }

        let aux_ev_rate = evaluate(data, data, &aux_weights, 0.5).2;
        calls_to_eval += 1;

        if aux_ev_rate > weights_and_result.result {
            weights_and_result.result = aux_ev_rate;
            weights_and_result.chromosome = aux_weights;
            fill_and_shuffle(&mut indices, rng);
        } else {
            if indices.is_empty() {
                fill_and_shuffle(&mut indices, rng);
            }
        }
    }

    calls_to_eval
}

pub fn generational_memetic_algorithm<T: DataElem<T> + Copy + Clone>(
    data: &Vec<T>,
    rng: &mut StdRng,
    population_size: usize,
    crossover_probability: f32,
    mutation_probability: f32,
    max_calls_to_eval: usize,
    memetic_type: usize,
    selection_operator: fn(
        &Vec<ChromosomeAndResult>,
        usize,
        &mut StdRng,
    ) -> Vec<ChromosomeAndResult>,
    crossover_operator: fn(&Chromosome, &Chromosome, &mut StdRng) -> (Chromosome, Chromosome),
    mutation_operator: fn(&mut Vec<f32>, usize, &mut StdRng),
) -> Vec<f32> {
    const DEBUG: bool = false;

    let quick_eval = |weights: &Chromosome| {
        return evaluate(data, data, weights, 0.5).2;
    };

    let mut generation_counter = 0;

    let num_gens_per_chromosome = T::get_num_attributes();
    let uniform = Uniform::new(0.0, 1.0);
    let mut calls_to_eval = 0;

    // Initialization ----
    let mut current_population: Vec<ChromosomeAndResult> = Vec::with_capacity(population_size);

    let mut aux_weights: Vec<f32> = Vec::with_capacity(T::get_num_attributes());
    // Initialize random weights (using normal distribution)
    for _ in 0..population_size {
        for _ in 0..num_gens_per_chromosome {
            aux_weights.push(uniform.sample(rng));
        }

        let aux_chrom_and_result = ChromosomeAndResult {
            chromosome: aux_weights.clone(),
            result: quick_eval(&aux_weights),
        };

        calls_to_eval += 1;

        current_population.push(aux_chrom_and_result);

        aux_weights.clear();
    }

    current_population.sort();
    let mut current_best_chrom_res = current_population.last().unwrap().clone();
    generation_counter += 1;

    while calls_to_eval < max_calls_to_eval {
        if false {
            println!("Generación número: {}", generation_counter);
            println!("Llamadas a eval: {}", calls_to_eval);
            println!("Mejor cromosoma: {}", current_best_chrom_res.result);
        }

        if DEBUG {
            for elem in current_population.iter() {
                println!("{}", elem.result);
            }
            println!("\n");
        }

        let mut auxiliar_population: Vec<ChromosomeAndResult> = Vec::with_capacity(population_size);

        // Selection  ----
        let parents = selection_operator(&current_population, population_size, rng);

        // Crossover (arithmetic mean) ----
        let num_of_crossovers = (crossover_probability * (population_size as f32) / 2.) as usize;

        if DEBUG {
            println!("Num of crossover: {}", num_of_crossovers);
            println!("Parents");
            for elem in parents.iter() {
                println!("{}", elem.result);
            }
        }

        // Adds children chromosomes
        for index in 0..num_of_crossovers {
            let children = crossover_operator(
                &parents[2 * index].chromosome,
                &parents[2 * index + 1].chromosome,
                rng,
            );

            auxiliar_population.push(ChromosomeAndResult::new(children.0, -1.));
            auxiliar_population.push(ChromosomeAndResult::new(children.1, -1.));
        }

        // Adds remaining chromosomes from parents
        for index in 2 * num_of_crossovers..population_size {
            auxiliar_population.push(parents[index].clone());
        }

        if DEBUG {
            println!("Childrens + Some Parents");
            for elem in auxiliar_population.iter() {
                println!("{}", elem.result);
            }
        }

        // Mutation ----
        let expected_num_of_mutations =
            mutation_probability * population_size as f32 * num_gens_per_chromosome as f32;
        let mut num_of_mutations = expected_num_of_mutations as usize;

        if rng.gen_range(0., 1.) < (expected_num_of_mutations - num_of_mutations as f32) {
            num_of_mutations += 1;
        }

        if DEBUG {
            println!("Num of mutations: {}", num_of_mutations);
        }

        for _ in 0..num_of_mutations {
            let selector = rng.gen_range(0, population_size * num_gens_per_chromosome);
            let chosen_chromosome = &mut auxiliar_population[selector % population_size].chromosome;
            let chosen_gene = selector / population_size;

            // Mutation operator
            mutation_operator(chosen_chromosome, chosen_gene, rng);

            auxiliar_population[selector % population_size].result = quick_eval(chosen_chromosome);
            calls_to_eval += 1;
        }

        // Evaluation ----
        for chrom_and_res in auxiliar_population.iter_mut() {
            if chrom_and_res.result == -1. {
                chrom_and_res.result = quick_eval(&chrom_and_res.chromosome);
                calls_to_eval += 1;
            }
        }

        auxiliar_population.sort();
        let new_best_chrom_res = auxiliar_population.last().unwrap().clone();

        if DEBUG {
            println!("New population(without replacing)");
            for elem in auxiliar_population.iter() {
                println!("{}", elem.result);
            }
        }

        // Replacement ----
        // If new best chrom is worst than the old one
        if new_best_chrom_res < current_best_chrom_res {
            auxiliar_population.remove(0);
            auxiliar_population.push(current_best_chrom_res.clone());
        } else {
            current_best_chrom_res = new_best_chrom_res;
        }

        current_population = auxiliar_population;

        generation_counter += 1;

        if generation_counter % 10 == 0 {
            match memetic_type {
                // Apply local search to every chromosome
                1 => {
                    for chrom_and_res in current_population.iter_mut() {
                        calls_to_eval += low_intensity_local_search(data, chrom_and_res, rng);
                    }
                }

                // Apply to a random percentage of the population
                // TODO choose_multiple
                2 => {
                    let local_seach_probability = 0.1;
                    let expected_local_searches = local_seach_probability * population_size as f32;
                    let mut num_local_searches = expected_local_searches as usize;

                    if rng.gen_range(0., 1.) < expected_local_searches - num_local_searches as f32 {
                        num_local_searches += 1;
                    }

                    for _ in 0..num_local_searches {
                        let index = rng.gen_range(0, population_size);
                        calls_to_eval +=
                            low_intensity_local_search(data, &mut current_population[index], rng);
                    }
                }
                3 => {
                    let num_local_searches = (0.1 * population_size as f32) as usize;

                    for ind in 0..num_local_searches {
                        calls_to_eval += low_intensity_local_search(
                            data,
                            &mut current_population[population_size - ind - 1],
                            rng,
                        );
                    }
                }

                _ => {}
            }

            current_population.sort();
            current_best_chrom_res = current_population.last().unwrap().clone();
        }
    }

    return current_best_chrom_res.chromosome;
}

pub fn simulated_annealing<T: DataElem<T> + Copy + Clone>(
    data: &Vec<T>,
    rng: &mut StdRng,
    max_calls_to_eval: usize,
    final_temperature: f32,
    phi: f32,
    mu: f32,
    mutation_operator: fn(&mut Vec<f32>, usize, &mut StdRng),
) -> Vec<f32> {
    let quick_eval = |weights: &Chromosome| {
        return evaluate(data, data, weights, 0.5).2;
    };

    let num_attributes = T::get_num_attributes();
    let uniform = Uniform::new(0.0, 1.0);

    // Initialize
    let mut current_solution: Vec<f32> = Vec::with_capacity(T::get_num_attributes());
    for _ in 0..num_attributes {
        current_solution.push(uniform.sample(rng));
    }
    let mut current_solution_fit = quick_eval(&current_solution);
    let mut best_solution = current_solution.clone();
    let mut best_solution_fit = current_solution_fit;

    // TODO Ask if this is correct (probably not)
    let max_neighbours = 10 * data.len();
    let max_successes = (0.1 * max_neighbours as f32) as usize;

    let num_coolings = max_calls_to_eval / max_neighbours;

    let initial_temperature = mu * (1. - best_solution_fit) / -phi.ln();

    let beta = (initial_temperature - final_temperature)
        / (initial_temperature * final_temperature * num_coolings as f32);

    let cool_down = |temperature: &f32| {
        return *temperature / (1. + beta * (*temperature));
    };

    let mut temperature = initial_temperature;
    let mut succeses = 0;

    for _ in 0..num_coolings {
        for _ in 0..max_neighbours {
            let aux_index = rng.gen_range(0, num_attributes);
            let mut possible_solution = current_solution.clone();
            mutation_operator(&mut possible_solution, aux_index, rng);
            let possible_sol_fit = quick_eval(&possible_solution);
            let diff = current_solution_fit - possible_sol_fit;

            if diff < 0. || uniform.sample(rng) <= (-1. * diff / temperature).exp() {
                current_solution = possible_solution;
                current_solution_fit = possible_sol_fit;

                if current_solution_fit >= best_solution_fit {
                    best_solution = current_solution.clone();
                    best_solution_fit = current_solution_fit;
                }

                succeses = succeses + 1;
                if succeses >= max_successes {
                    break;
                }
            }
        }

        if succeses == 0 {
            break;
        } else {
            succeses = 0;
        }

        temperature = cool_down(&temperature);
    }

    return best_solution;
}

pub fn iterated_local_search<T: DataElem<T> + Copy + Clone>(
    data: &Vec<T>,
    rng: &mut StdRng,
    calls_to_local_search: usize,
    weight_mutation_probability: f32,
    mutation_operator: fn(&mut Vec<f32>, usize, &mut StdRng),
    local_search: fn(&Vec<T>, &mut StdRng, &Vec<f32>) -> Vec<f32>,
) -> Vec<f32> {
    let quick_eval = |weights: &Vec<f32>| {
        return evaluate(data, data, weights, 0.5).2;
    };

    let num_attributes = T::get_num_attributes();
    let expected_num_of_mutations = weight_mutation_probability * num_attributes as f32;

    let mut best_solution: Vec<f32> = random_weights_generator(num_attributes, rng);
    best_solution = local_search(data, rng, &best_solution);
    let mut best_solution_fit = quick_eval(&best_solution);

    for _ in 0..calls_to_local_search - 1 {
        let mut possible_solution = best_solution.clone();

        let mut num_of_mutations = expected_num_of_mutations as usize;
        if rng.gen_range(0., 1.) < (expected_num_of_mutations - num_of_mutations as f32) {
            num_of_mutations += 1;
        }

        let selected_indices = (0..num_attributes).choose_multiple(rng, num_of_mutations);
        for index in selected_indices {
            mutation_operator(&mut possible_solution, index, rng);
        }

        possible_solution = local_search(data, rng, &possible_solution);
        let possible_solution_fit = quick_eval(&possible_solution);

        if possible_solution_fit > best_solution_fit {
            best_solution = possible_solution;
            best_solution_fit = possible_solution_fit;
        }
    }

    best_solution
}

pub fn local_search_max_calls<T: DataElem<T> + Copy + Clone>(
    data: &Vec<T>,
    rng: &mut StdRng,
    initial_weights: &Vec<f32>,
    max_calls_to_eval: usize,
) -> Vec<f32> {
    let num_attrs = T::get_num_attributes();
    let mut calls_to_eval = 0;

    // Closure that fills and shuffle indices
    let fill_and_shuffle = |indices: &mut Vec<usize>, rng: &mut StdRng| {
        *indices = (0..num_attrs).collect();
        indices.shuffle(rng);
    };

    // Initialize vector of indices and shuffles it
    let mut indices: Vec<usize> = Vec::with_capacity(num_attrs);
    fill_and_shuffle(&mut indices, rng);

    // Normal distribution with mean = 0.0, standard deviation = 0.3
    let normal = Normal::new(0.0, 0.3);

    let mut best_weights = initial_weights.clone();
    let mut best_weights_fitness = evaluate(data, data, &best_weights, 0.5).2;

    while calls_to_eval < max_calls_to_eval {
        let mut aux_weights = best_weights.clone();

        let index = indices.pop().expect("El vector está vacio");

        // Mutation
        aux_weights[index] += normal.sample(rng) as f32;

        // Truncate into [0,1]
        if aux_weights[index] < 0. {
            aux_weights[index] = 0.;
        } else if aux_weights[index] > 1. {
            aux_weights[index] = 1.;
        }

        let aux_ev_rate = evaluate(data, data, &aux_weights, 0.5).2;
        calls_to_eval += 1;

        if aux_ev_rate > best_weights_fitness {
            best_weights_fitness = aux_ev_rate;
            best_weights = aux_weights;
            fill_and_shuffle(&mut indices, rng);
        } else {
            if indices.is_empty() {
                fill_and_shuffle(&mut indices, rng);
            }
        }
    }

    best_weights
}

// Diferential evolution rand 1
pub fn de_rand1<T: DataElem<T> + Copy + Clone>(
    data: &Vec<T>,
    rng: &mut StdRng,
    max_calls_to_eval: usize,
    population_size: usize,
    crossover_probability: f32,
    f: f32,
) -> Vec<f32> {
    let quick_eval = |weights: &Vec<f32>| {
        return evaluate(data, data, weights, 0.5).2;
    };

    let mut calls_to_eval = 0;
    let mut population: Vec<ChromosomeAndResult> = Vec::with_capacity(population_size);
    let num_attributes = T::get_num_attributes();

    for _ in 0..population_size {
        let aux = random_weights_generator(num_attributes, rng);
        population.push(ChromosomeAndResult {
            chromosome: aux.clone(),
            result: quick_eval(&aux),
        });
        calls_to_eval += 1;
    }

    while calls_to_eval < max_calls_to_eval {
        let mut offspring_population: Vec<ChromosomeAndResult> =
            Vec::with_capacity(population_size);
        for i in 0..population_size {
            let mut offspring: Vec<f32> = Vec::with_capacity(num_attributes);

            // Select parents indices
            let mut parents_indices = (0..population_size).choose_multiple(rng, 3);
            while parents_indices.contains(&i) {
                parents_indices = (0..population_size).choose_multiple(rng, 3);
            }

            let parent1 = &population[parents_indices[0]].chromosome;
            let parent2 = &population[parents_indices[1]].chromosome;
            let parent3 = &population[parents_indices[2]].chromosome;

            let rand_gene = rng.gen_range(0, num_attributes);

            for gene in 0..num_attributes {
                if rng.gen_range(0., 1.) < crossover_probability || gene == rand_gene {
                    offspring.push(truncate_01(
                        &(parent1[gene] + f * (parent2[gene] - parent3[gene])),
                    ))
                } else {
                    offspring.push(population[i].chromosome[gene]);
                }
            }

            offspring_population.push(ChromosomeAndResult {
                chromosome: offspring.clone(),
                result: quick_eval(&offspring),
            });

            calls_to_eval += 1;
        }

        for i in 0..population_size {
            if population[i] < offspring_population[i] {
                population[i] = offspring_population[i].clone();
            }
        }
    }

    population.sort();

    return population
        .last()
        .expect("Población vacía en evolución diferencial")
        .chromosome
        .clone();
}

// Diferential evolution
pub fn de_current_to_best<T: DataElem<T> + Copy + Clone>(
    data: &Vec<T>,
    rng: &mut StdRng,
    max_calls_to_eval: usize,
    population_size: usize,
    crossover_probability: f32,
    f: f32,
) -> Vec<f32> {
    let quick_eval = |weights: &Vec<f32>| {
        return evaluate(data, data, weights, 0.5).2;
    };

    let mut calls_to_eval = 0;
    let mut population: Vec<ChromosomeAndResult> = Vec::with_capacity(population_size);
    let num_attributes = T::get_num_attributes();

    for _ in 0..population_size {
        let aux = random_weights_generator(num_attributes, rng);
        population.push(ChromosomeAndResult {
            chromosome: aux.clone(),
            result: quick_eval(&aux),
        });
        calls_to_eval += 1;
    }

    while calls_to_eval < max_calls_to_eval {
        population.sort();
        let best = population
            .last()
            .expect("Empty population in de_current_to_best")
            .chromosome
            .clone();
        let mut offspring_population: Vec<ChromosomeAndResult> =
            Vec::with_capacity(population_size);
        for i in 0..population_size {
            let current = &population[i].chromosome;
            let mut offspring: Vec<f32> = Vec::with_capacity(num_attributes);

            // Select parents indices
            let parents_indices = (0..population_size).choose_multiple(rng, 2);
            let parent1 = &population[parents_indices[0]].chromosome;
            let parent2 = &population[parents_indices[1]].chromosome;

            let rand_gene = rng.gen_range(0, population_size);

            for gene in 0..num_attributes {
                if rng.gen_range(0., 1.) < crossover_probability || gene == rand_gene {
                    offspring.push(truncate_01(
                        &(current[gene]
                            + f * (best[gene] - current[gene])
                            + f * (parent1[gene] - parent2[gene])),
                    ));
                } else {
                    offspring.push(population[i].chromosome[gene]);
                }
            }

            offspring_population.push(ChromosomeAndResult {
                chromosome: offspring.clone(),
                result: quick_eval(&offspring),
            });

            calls_to_eval += 1;
        }

        for i in 0..population_size {
            if population[i] < offspring_population[i] {
                population[i] = offspring_population[i].clone();
            }
        }
    }

    population.sort();

    return population
        .last()
        .expect("Población vacía en evolución diferencial")
        .chromosome
        .clone();
}

// Operators for genetic algorithm -------
pub fn binary_tournament_selection(
    population: &Vec<ChromosomeAndResult>,
    num_parents: usize,
    rng: &mut StdRng,
) -> Vec<ChromosomeAndResult> {
    let population_size = population.len();
    let mut parents: Vec<ChromosomeAndResult> = Vec::with_capacity(num_parents);

    for _ in 0..num_parents {
        let selector = rng.gen_range(0, population_size * population_size);
        let first_competitor = &population[selector / population_size];
        let second_competitor = &population[selector % population_size];

        if first_competitor > second_competitor {
            parents.push((*first_competitor).clone());
        } else {
            parents.push((*second_competitor).clone());
        }
    }

    parents
}

pub fn arithmetic_mean_crossover(
    one: &Chromosome,
    other: &Chromosome,
    alpha: f32,
) -> (Chromosome, Chromosome) {
    let mut chromosomes: (Chromosome, Chromosome) =
        (Vec::with_capacity(one.len()), Vec::with_capacity(one.len()));
    for gene in 0..one.len() {
        chromosomes
            .0
            .push(one[gene] * (alpha) + other[gene] * (1. - alpha));
        chromosomes
            .1
            .push(one[gene] * (1. - alpha) + other[gene] * (alpha));
    }

    return chromosomes;
}

pub fn blx_alpha_crossover(
    one: &Chromosome,
    other: &Chromosome,
    alpha: f32,
    rng: &mut StdRng,
) -> (Chromosome, Chromosome) {
    let mut chromosomes: (Chromosome, Chromosome) =
        (Vec::with_capacity(one.len()), Vec::with_capacity(one.len()));
    for gene in 0..one.len() {
        let c_max;
        let c_min;

        if one[gene] < other[gene] {
            c_max = other[gene];
            c_min = one[gene];
        } else if other[gene] < one[gene] {
            c_max = one[gene];
            c_min = other[gene];
        } else {
            chromosomes.0.push(one[gene]);
            chromosomes.1.push(one[gene]);

            continue;
        }

        let dist = c_max - c_min;

        let mut gene_value1 = rng.gen_range(c_min - dist * alpha, c_max + dist * alpha);
        let mut gene_value2 = rng.gen_range(c_min - dist * alpha, c_max + dist * alpha);

        // Truncate into [0,1] gene_value1
        if gene_value1 < 0. {
            gene_value1 = 0.;
        } else if gene_value1 > 1. {
            gene_value1 = 1.;
        }

        // Truncate into [0,1] gene_value2
        if gene_value2 < 0. {
            gene_value2 = 0.;
        } else if gene_value2 > 1. {
            gene_value2 = 1.;
        }

        chromosomes.0.push(gene_value1);
        chromosomes.1.push(gene_value2);
    }

    chromosomes
}

pub fn weight_mutation(
    weights: &mut Vec<f32>,
    index: usize,
    mean: f32,
    std_deviation: f32,
    rng: &mut StdRng,
) {
    // Normal distribution with determined mean and standard deviation
    let normal = Normal::new(mean.into(), std_deviation.into());

    weights[index] += normal.sample(rng) as f32;
    weights[index] = truncate_01(&weights[index]);
}

//---------------------------------------------------------------------------------------
// Evaluation function for results
//---------------------------------------------------------------------------------------

/// Calculates the percentage of correctly classified elems
///
/// # Arguments
///
/// * `data` - Vec of `T` used as knowledge
/// * `guessing` - Vec of i32 which represents the guessed class for every elem of data
///
/// # Returns
/// A `Result` with:
/// * `Err(&'static str)` if the size of `guessing` doesn't coincide with the size of `data`
/// * `Ok(f32)` with the percentage

pub fn class_rate<T: DataElem<T> + Copy + Clone>(
    data: &Vec<T>,
    guessing: &Vec<i32>,
) -> Result<f32, &'static str> {
    if data.len() != guessing.len() {
        return Err("El numero de clases no coincide con el numero de elementos");
    }

    let mut counter = 0.0;

    for (index, elem) in data.iter().enumerate() {
        if elem.get_class() == guessing[index] {
            counter += 1.0;
        }
    }

    return Ok(100.0 * counter / (data.len() as f32));
}

/// Calculates the percentage of attributes that are not going to be used because
/// its weights are too low (under 0.2)
///
/// # Arguments
///
/// * `weights` - Vec of f32 representing weights
///
/// # Returns
/// A f32 with the percentage

pub fn red_rate(weights: &Vec<f32>) -> f32 {
    let mut counter = 0.0;

    for w in weights.iter() {
        if *w < 0.2 {
            counter += 1.0;
        }
    }

    return 100.0 * counter / (weights.len() as f32);
}

/// Calculates the evaluation function using classification and reduction rates
/// previously calculated
/// # Arguments
///
/// * `class_rate` - Classification rate
/// * `red_rate` - Reduction rate
/// * `alpha` - Value of classification importance over reduction (between 0 and 1)
///
/// # Returns
/// An f32 with the result

pub fn evaluation_function(class_rate: f32, red_rate: f32, alpha: f32) -> f32 {
    return alpha * class_rate + (1.0 - alpha) * red_rate;
}

//---------------------------------------------------------------------------------------
// Full algorithms execution functions
//---------------------------------------------------------------------------------------

// Random --------------
pub fn alg_random<T: DataElem<T> + Copy + Clone>(
    training_set: &Vec<T>,
    test_set: &Vec<T>,
    rng: &mut StdRng,
) -> (f32, f32, f32) {
    let mut guessing: Vec<i32> = Vec::new();

    for _ in test_set.iter() {
        let random_elem = training_set[rng.gen_range(0, training_set.len())];
        guessing.push(random_elem.get_class());
    }

    // Results
    let c_rate: f32 = class_rate(test_set, &guessing)
        .expect("No coincide el numero de elementos del test con el numero de <<guessings>>");
    let r_rate: f32 = 100.;
    let ev_rate: f32 = evaluation_function(c_rate, r_rate, 0.5);

    return (c_rate, r_rate, ev_rate);
}

pub fn alg_random_weights<T: DataElem<T> + Copy + Clone>(
    training_set: &Vec<T>,
    test_set: &Vec<T>,
    rng: &mut StdRng,
) -> (f32, f32, f32) {
    let mut guessing: Vec<i32> = Vec::new();
    let mut weights: Vec<f32> = Vec::with_capacity(T::get_num_attributes());

    for _ in 0..T::get_num_attributes() {
        weights.push(rng.gen_range(0., 1.));
    }

    for elem in test_set.iter() {
        guessing.push(
            classifier_1nn_with_weights(training_set, elem, &weights)
                .expect("Classifier 1nn en alg_random_weights"),
        );
    }

    // Results
    let c_rate: f32 = class_rate(test_set, &guessing)
        .expect("No coincide el numero de elementos del test con el numero de <<guessings>>");
    let r_rate: f32 = red_rate(&weights);
    let ev_rate: f32 = evaluation_function(c_rate, r_rate, 0.5);

    return (c_rate, r_rate, ev_rate);
}

pub fn evaluate<T: DataElem<T> + Copy + Clone>(
    training_set: &Vec<T>,
    test_set: &Vec<T>,
    weights: &Vec<f32>,
    alpha: f32,
) -> (f32, f32, f32) {
    let mut guessing: Vec<i32> = Vec::new();

    for elem in test_set.iter() {
        guessing.push(
            classifier_1nn_with_weights(&training_set, elem, &weights)
                .expect("No coincide el número de pesos  con el de atributos"),
        );
    }

    let c_rate: f32 = class_rate(test_set, &guessing)
        .expect("No coincide el numero de elementos del test con el numero de <<guessings>>");
    let r_rate: f32 = red_rate(&weights);
    let ev_rate = evaluation_function(c_rate, r_rate, alpha);

    return (c_rate, r_rate, ev_rate);
}

// ------------------------------------------------------------------------------------

/// Runs every algorithm using the data set passed as parameter
/// and prints results.
/// Precondition: Generic tipe must coincide with file
///
/// # Arguments
///
/// * `file_name` - Name of file containing data
///
/// # Returns
/// A Result that contains the error in case it fails.
pub fn run<T: DataElem<T> + Copy + Clone>(
    file_name: &str,
    rng: &mut StdRng,
) -> Result<(), Box<Error>> {
    let mut data: Vec<T> = Vec::new();
    let mut rdr = csv::Reader::from_path(file_name)?;

    let mut current_id = 0;
    for result in rdr.records() {
        let mut aux_record = T::new();
        let record = result?;

        let mut counter = 0;

        aux_record.set_id(current_id);

        for field in record.iter() {
            // CSV structure: ... 40 data ... , class
            if counter != T::get_num_attributes() {
                aux_record.set_attribute(counter, field.parse::<f32>().unwrap());
            } else {
                aux_record.set_class(field.parse::<i32>().unwrap());
            }

            counter += 1;
        }

        current_id += 1;

        data.push(aux_record);
    }

    data = normalize_data(data);

    let partitions = make_partitions(&data);

    let mut weights_generators: Vec<fn(&Vec<T>, &mut StdRng) -> Vec<f32>> = Vec::new();
    let mut weights_generators_names: Vec<&str> = Vec::new();

    // // 1nn
    // weights_generators
    //     .push(|training_set: &Vec<T>, rng: &mut StdRng| vec![1.; T::get_num_attributes()]);
    // weights_generators_names.push("1-NN");
    // // Relief algorithm
    // weights_generators
    //     .push(|training_set: &Vec<T>, rng: &mut StdRng| relief_algorithm(training_set));
    // weights_generators_names.push("Relief");
    // // Local search algorithm
    // weights_generators.push(|training_set: &Vec<T>, rng: &mut StdRng| {
    //     local_search(
    //         training_set,
    //         rng,
    //         |weights: &mut Vec<f32>, index: usize, rng: &mut StdRng| {
    //            weight_mutation(weights, index, 0.0, 0.3, rng)
    //         },
    //         15000,
    //         20 * T::get_num_attributes(),
    //     )
    // });
    // weights_generators_names.push("Local Search");

    // // AGG - Arithmetic_0.4
    // weights_generators.push(|training_set: &Vec<T>, rng: &mut StdRng| {
    //     generational_genetic_algorithm(
    //         training_set,
    //         rng,
    //         30,
    //         0.7,
    //         0.001,
    //         15000,
    //         binary_tournament_selection,
    //         |ch1: &Chromosome, ch2: &Chromosome, _rng: &mut StdRng| {
    //             arithmetic_mean_crossover(ch1, ch2, 0.4)
    //         },
    //         |weights: &mut Vec<f32>, index: usize, rng: &mut StdRng| {
    //             weight_mutation(weights, index, 0.0, 0.3, rng)
    //         },
    //     )
    // });
    // weights_generators_names.push("AGG - Arithmetic_0.4");

    // // AGG - BLX_0.3
    // weights_generators.push(|training_set: &Vec<T>, rng: &mut StdRng| {
    //     generational_genetic_algorithm(
    //         training_set,
    //         rng,
    //         30,
    //         0.7,
    //         0.001,
    //         15000,
    //         binary_tournament_selection,
    //         |ch1: &Chromosome, ch2: &Chromosome, rng: &mut StdRng| {
    //             blx_alpha_crossover(ch1, ch2, 0.3, rng)
    //         },
    //         |weights: &mut Vec<f32>, index: usize, rng: &mut StdRng| {
    //             weight_mutation(weights, index, 0.0, 0.3, rng)
    //         },
    //     )
    // });
    // weights_generators_names.push("AGG - BLX_0.3");

    // // AGE - Arithmetic_0.4
    // weights_generators.push(|training_set: &Vec<T>, rng: &mut StdRng| {
    //     steady_state_genetic_algorithm(
    //         training_set,
    //         rng,
    //         30,
    //         0.001,
    //         15000,
    //         binary_tournament_selection,
    //         |ch1: &Chromosome, ch2: &Chromosome, _: &mut StdRng| {
    //             arithmetic_mean_crossover(ch1, ch2, 0.4)
    //         },
    //         |weights: &mut Vec<f32>, index: usize, rng: &mut StdRng| {
    //             weight_mutation(weights, index, 0.0, 0.3, rng)
    //         },
    //     )
    // });
    // weights_generators_names.push("AGE - Arithmetic_0.4");

    // // AGE - BLX_0.3
    // weights_generators.push(|training_set: &Vec<T>, rng: &mut StdRng| {
    //     steady_state_genetic_algorithm(
    //         training_set,
    //         rng,
    //         30,
    //         0.001,
    //         15000,
    //         binary_tournament_selection,
    //         |ch1: &Chromosome, ch2: &Chromosome, rng: &mut StdRng| {
    //             blx_alpha_crossover(ch1, ch2, 0.3, rng)
    //         },
    //         |weights: &mut Vec<f32>, index: usize, rng: &mut StdRng| {
    //             weight_mutation(weights, index, 0.0, 0.3, rng)
    //         },
    //     )
    // });
    // weights_generators_names.push("AGE - BLX_0.3");

    // // AM - 1
    // weights_generators.push(|training_set: &Vec<T>, rng: &mut StdRng| {
    //     generational_memetic_algorithm(
    //         training_set,
    //         rng,
    //         10,
    //         0.7,
    //         0.001,
    //         15000,
    //         1,
    //         binary_tournament_selection,
    //         |ch1: &Chromosome, ch2: &Chromosome, rng: &mut StdRng| {
    //             blx_alpha_crossover(ch1, ch2, 0.3, rng)
    //         },
    //         |weights: &mut Vec<f32>, index: usize, rng: &mut StdRng| {
    //             weight_mutation(weights, index, 0.0, 0.3, rng)
    //         },
    //     )
    // });
    // weights_generators_names.push("AM - 1");

    // // AM - 0.1
    // weights_generators.push(|training_set: &Vec<T>, rng: &mut StdRng| {
    //     generational_memetic_algorithm(
    //         training_set,
    //         rng,
    //         10,
    //         0.7,
    //         0.001,
    //         15000,
    //         2,
    //         binary_tournament_selection,
    //         |ch1: &Chromosome, ch2: &Chromosome, rng: &mut StdRng| {
    //             blx_alpha_crossover(ch1, ch2, 0.3, rng)
    //         },
    //         |weights: &mut Vec<f32>, index: usize, rng: &mut StdRng| {
    //             weight_mutation(weights, index, 0.0, 0.3, rng)
    //         },
    //     )
    // });
    // weights_generators_names.push("AM - 0.1");

    // // AM - 0.1best
    // weights_generators.push(|training_set: &Vec<T>, rng: &mut StdRng| {
    //     generational_memetic_algorithm(
    //         training_set,
    //         rng,
    //         10,
    //         0.7,
    //         0.001,
    //         15000,
    //         3,
    //         binary_tournament_selection,
    //         |ch1: &Chromosome, ch2: &Chromosome, rng: &mut StdRng| {
    //             blx_alpha_crossover(ch1, ch2, 0.3, rng)
    //         },
    //         |weights: &mut Vec<f32>, index: usize, rng: &mut StdRng| {
    //             weight_mutation(weights, index, 0.0, 0.3, rng)
    //         },
    //     )
    // });
    // weights_generators_names.push("AM - 0.1best");

    // // Simulated Annealing
    // weights_generators.push(|training_set: &Vec<T>, rng: &mut StdRng| {
    //     simulated_annealing(
    //         training_set,
    //         rng,
    //         15000,
    //         0.001,
    //         0.3,
    //         0.3,
    //         |weights: &mut Vec<f32>, index: usize, rng: &mut StdRng| {
    //             weight_mutation(weights, index, 0.0, 0.3, rng)
    //         },
    //     )
    // });
    // weights_generators_names.push("Simulated Annealing");

    // Iterated local search
    weights_generators.push(|training_set: &Vec<T>, rng: &mut StdRng| {
        iterated_local_search(
            training_set,
            rng,
            15,
            0.1,
            |weights: &mut Vec<f32>, index: usize, rng: &mut StdRng| {
                weight_mutation(weights, index, 0.0, 0.4, rng)
            },
            |data: &Vec<T>, rng: &mut StdRng, initial_weights: &Vec<f32>| {
                local_search(
                    data,
                    rng,
                    initial_weights,
                    |weights: &mut Vec<f32>, index: usize, rng: &mut StdRng| {
                        weight_mutation(weights, index, 0.0, 0.3, rng)
                    },
                    1000,
                    20 * T::get_num_attributes(),
                )
            },
        )
    });
    weights_generators_names.push("Iterated LS");

    // // Diferential Evolution Rand1
    // weights_generators.push(|training_set: &Vec<T>, rng: &mut StdRng| {
    //     de_rand1(training_set, rng, 15000, 50, 0.5, 0.5)
    // });
    // weights_generators_names.push("Diferential Evolution Rand1");

    // // Diferential Evolution Current Best
    // weights_generators.push(|training_set: &Vec<T>, rng: &mut StdRng| {
    //     de_current_to_best(training_set, rng, 15000, 50, 0.5, 0.5)
    // });
    // weights_generators_names.push("Diferential Evolution Current Best");

    let mut result_tables =
        vec![
            table!(["Partición", "Tasa_clas", "Tasa_red", "Agregado", "Tiempo"]);
            weights_generators.len()
        ];

    for test in 0..5 {
        // Stablish training and test sets
        let mut training_set: Vec<T> = Vec::new();
        let mut test_set: Vec<T> = Vec::new();

        for part in 0..5 {
            if part != test {
                training_set.extend(&partitions[part]);
            } else {
                test_set = partitions[part].clone();
            }
        }

        let mut now;

        for (ind, weights_generator) in weights_generators.iter().enumerate() {
            println!("Ejecutando {}", weights_generators_names[ind]);
            now = Instant::now();
            let weights = weights_generator(&training_set, rng);
            let results = evaluate(&training_set, &test_set, &weights, 0.5);

            let time_elapsed = now.elapsed().as_millis();

            result_tables[ind].add_row(row![test, results.0, results.1, results.2, time_elapsed,]);
        }
    }

    for (ind, table) in result_tables.iter().enumerate() {
        println!(
            "Resultados para el algoritmo {}",
            weights_generators_names[ind]
        );
        table.printstd();
    }

    Ok(())
}

fn main() {
    let args: Vec<_> = env::args().collect();

    let mut seed: u64 = 1;
    if args.len() == 2 {
        seed = args[1].parse::<u64>().unwrap();
    } else if args.len() > 2 {
        println!("* Ningún argumento -> semilla por defecto (1)\n* Un argumento -> ese argumento como semilla");
        process::exit(1);
    }

    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);

    println!("--------------");
    println!("| Texture    |");
    println!("--------------");
    if let Err(err) = run::<TextureRecord>("data/texture.csv", &mut rng) {
        println!("error en texture: {}", err);
        process::exit(1);
    }

    println!("--------------");
    println!("| Colposcopy |");
    println!("--------------");
    if let Err(err) = run::<ColposcopyRecord>("data/colposcopy.csv", &mut rng) {
        println!("error en colposcopy: {}", err);
        process::exit(1);
    }

    println!("--------------");
    println!("| Ionosphere |");
    println!("--------------");
    if let Err(err) = run::<IonosphereRecord>("data/ionosphere.csv", &mut rng) {
        println!("error ionosphere: {}", err);
        process::exit(1);
    }
}
