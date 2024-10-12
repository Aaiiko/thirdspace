import React from "react";

interface BusinessCardProps {
    business: {
        name: string;
        description: string;
        [key: string]: any;
    };
}

export const BusinessCard: React.FC<BusinessCardProps> = ({ business }) => {
    return (
        <div>
            <h2 className="text-xl font-bold">{business.name}</h2>
            <p>{business.description}</p>
        </div>
    );
};