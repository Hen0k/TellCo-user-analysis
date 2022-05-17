CREATE TABLE IF NOT EXISTS `tellco_users_scores` 
(
    `id` INT NOT NULL AUTO_INCREMENT,
    `msisdn` TEXT NOT NULL,
    `satisfaction_score` FLOAT DEFAULT NULL,
    `engagement_score` FLOAT DEFAULT NULL,
    `experiance_score` FLOAT DEFAULT NULL,
    PRIMARY KEY (`id`)
);
